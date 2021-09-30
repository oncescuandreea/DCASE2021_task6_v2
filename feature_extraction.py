#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch
import csv
import torch.nn as nn
import time
import sys
from loguru import logger
import argparse
from tqdm import tqdm
from pathlib import Path
from data_handling.clotho_dataset import get_clotho_loader
from data_handling.audiocaps_dataset import get_audiocaps_loader
from data_handling.sounddescs_dataset import get_sounddescs_loader
from data_handling.test_dataset import get_test_loader
from tools.config_loader import get_config
from tools.utils import setup_seed, align_word_embedding, \
LabelSmoothingLoss, set_tgt_padding_mask, rotation_logger, \
decode_output, beam_search, greedy_decode, mixup_data
from tools.file_io import load_picke_file
from models.TransModel import TransformerModel
from pprint import PrettyPrinter
from warmup_scheduler import GradualWarmupScheduler
from eval_metrics import evaluate_metrics
import numpy as np
import pickle


def extract_features(data_split: torch.utils.data.DataLoader, feats_dir: Path):
    model.eval()
    with torch.no_grad():
        for batch_idx, eval_batch in tqdm(enumerate(data_split), total=len(data_split)):
            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            encoder_output = model.feature_extractor.intermediate_forward(src)
            encoder_output_mean = torch.mean(encoder_output, 0).cpu()
            for idx, file_name in enumerate(f_names):
                featdict = dict()
                file_feats = np.array(encoder_output_mean[idx, :], dtype=np.float16)
                featdict['feats'] = file_feats.reshape(1, 512)
                # if (feats_dir / f'{file_name.split(".wav")[0]}.pkl').exists() is False:
                with open(feats_dir / f'{file_name.split(".wav")[0]}.pkl', 'wb') as f:
                    pickle.dump(featdict, f)


parser = argparse.ArgumentParser(description='Settings for audio caption model')

parser.add_argument('-n', '--exp_name', type=str, default='exp_feats', help='name of the experiment')
parser.add_argument('-m', '--mask_type', type=str, default='zero_value', help='masking type')

args = parser.parse_args()

config = get_config('settings_feat.yaml')

config.training.spec_type = args.mask_type

setup_seed(config.training.seed)

exp_name = args.exp_name

# output setting
model_output_dir = Path('outputs', exp_name, 'model')
log_output_dir = Path('outputs', exp_name, 'logging')

model_output_dir.mkdir(parents=True, exist_ok=True)
log_output_dir.mkdir(parents=True, exist_ok=True)

logger.remove()

logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}',
           level='INFO', filter=lambda record: record['extra']['indent'] == 1)

logger.add(log_output_dir.joinpath('output.txt'),
           format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 1)

logger.add(str(log_output_dir) + '/captions.txt',
           format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 2,
           rotation=rotation_logger)

logger.add(str(log_output_dir) + '/beam_captions.txt',
           format='{message}', level='INFO',
           filter=lambda record: record['extra']['indent'] == 3,
           rotation=rotation_logger)

main_logger = logger.bind(indent=1)

printer = PrettyPrinter()

device, device_name = (torch.device('cuda'), torch.cuda.get_device_name(torch.cuda.current_device()))

main_logger.info(f'Process on {device_name}')

dataset = config.feature_data.type
print(f'dataset used is {dataset}')
dataset_dict = {'audiocaps': 'AudioCaps', 'clotho': 'CLOTHO', 'sounddescs': 'BBCSound'}
feats_dir = Path(f'/scratch/shared/beegfs/oncescu/shared-datasets/{dataset_dict[dataset]}/processing/pred_audiocaps/audiocaps')
feats_dir.mkdir(parents=True, exist_ok=True)

batch_size = config.data.batch_size
num_workers = config.data.num_workers
input_field_name = config.data.input_field_name

# data loading
if dataset == 'clotho':
    # words_list_path = config.path.clotho.words_list
    # words_freq_path = config.path.clotho.words_freq
    training_data = get_clotho_loader(split='development',
                                      input_field_name=input_field_name,
                                      load_into_memory=False,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=num_workers)

    validation_data = get_clotho_loader(split='validation',
                                        input_field_name=input_field_name,
                                        load_into_memory=False,
                                        batch_size=batch_size,
                                        num_workers=num_workers)

    evaluation_data = get_clotho_loader(split='evaluation',
                                        input_field_name=input_field_name,
                                        load_into_memory=False,
                                        batch_size=batch_size,
                                        num_workers=num_workers)
elif dataset == 'audiocaps':
    # words_list_path = config.path.audiocaps.words_list
    # words_freq_path = config.path.audiocaps.words_freq
    training_data = get_audiocaps_loader(split='train',
                                         batch_size=batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=num_workers)

    validation_data = get_audiocaps_loader(split='val',
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    evaluation_data = get_audiocaps_loader(split='test',
                                           batch_size=batch_size,
                                           num_workers=num_workers)
elif dataset == 'sounddescs':
    # words_list_path = config.path.audiocaps.words_list
    # words_freq_path = config.path.audiocaps.words_freq
    training_data = get_sounddescs_loader(split='train',
                                         batch_size=batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=num_workers)

    validation_data = get_sounddescs_loader(split='val',
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    evaluation_data = get_sounddescs_loader(split='test',
                                           batch_size=batch_size,
                                           num_workers=num_workers)

# loading vocabulary list
if config.mode == 'finetune' and config.finetune.audiocap:
    words_list_path = 'data/pickles/new_words_list.p'
if config.data.type == 'audiocaps':
    words_list_path = config.path.audiocaps.words_list
elif config.data.type == 'clotho':
    words_list_path = config.path.clotho.words_list
words_list = load_picke_file(words_list_path)
ntokens = len(words_list)
sos_ind = words_list.index('<sos>')
eos_ind = words_list.index('<eos>')

pretrained_cnn = torch.load(config.path.encoder + config.encoder.model + '.pth')['model'] if config.encoder.pretrained else None

pretrained_word_embedding = align_word_embedding(words_list, config.path.word2vec, config.decoder.nhid) if config.word_embedding.pretrained else None


main_logger.info('Training setting:\n'
                 f'{printer.pformat(config)}')

model = TransformerModel(config, words_list, pretrained_cnn, pretrained_word_embedding)

model.to(device)


main_logger.info(f'Model:\n{model}\n')
main_logger.info('Total number of parameters:'
                 f'{sum([i.numel() for i in model.parameters()])}')

main_logger.info(f'Len of training data: {len(training_data)}')
main_logger.info(f'Len of evaluation data: {len(evaluation_data)}')


eval_spiders = []
eval_metrics = {}
main_logger.info('Evaluation mode.')

model.load_state_dict(torch.load(config.path.model)['model'])
main_logger.info('Metrcis on evaluation set')
# import pdb; pdb.set_trace()
data_splits = [training_data, validation_data, evaluation_data]
for data_split in data_splits:
    extract_features(data_split, feats_dir)

main_logger.info('Evaluation done.')
for metric, values in eval_metrics.items():
    main_logger.info(f'best metric: {metric:<7s}: {values["score"]:7.4f}')

