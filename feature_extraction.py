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


def eval_greedy(data, max_len=30):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_all = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            output = greedy_decode(model, src, sos_ind=sos_ind)

            output = output[:, 1:].int()
            y_hat_batch = torch.zeros(output.shape).fill_(eos_ind).to(device)

            for i in range(output.shape[0]):    # batch_size
                for j in range(output.shape[1]):
                    y_hat_batch[i, j] = output[i, j]
                    if output[i, j] == eos_ind:
                        break

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_all.extend(captions)
            file_names_all.extend(f_names)

        end_time = time.time()
        eval_time = end_time - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_all, file_names_all, words_list, log_output_dir)
        greedy_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = greedy_metrics['spider']['score']
        cider = greedy_metrics['cider']['score']
        main_logger.info(f'Cider: {cider:7.4f}')
        main_logger.info(f'Spider score using greedy search: {spider:7.4f}, eval time: {eval_time:.4f}')


def eval_beam(data, beam_size, max_len=30):

    model.eval()
    with torch.no_grad():
        start_time = time.time()
        y_hat_all = []
        ref_captions_all = []
        file_names_all = []

        for batch_idx, eval_batch in tqdm(enumerate(data), total=len(data)):

            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            output = beam_search(model, src, sos_ind=sos_ind, eos_ind=eos_ind, beam_size=beam_size)

            y_hat_batch = torch.zeros([src.shape[0], max_len]).fill_(eos_ind).to(device)

            for i, o in enumerate(output):    # batch_size
                o = o[1:]
                for j in range(max_len - 1):
                    y_hat_batch[i, j] = o[j]
                    if o[j] == eos_ind:
                        break

            y_hat_batch = y_hat_batch.int()
            y_hat_all.extend(y_hat_batch.cpu())
            ref_captions_all.extend(captions)
            file_names_all.extend(f_names)

        end_time = time.time()
        eval_time = end_time - start_time
        captions_pred, captions_gt = decode_output(y_hat_all, ref_captions_all, file_names_all, words_list, log_output_dir, beam=True)
        beam_metrics = evaluate_metrics(captions_pred, captions_gt)
        spider = beam_metrics['spider']['score']
        cider = beam_metrics['cider']['score']
        main_logger.info(f'Cider: {cider:7.4f}')
        main_logger.info(f'Spider score using beam search (beam size:{beam_size}): {spider:7.4f}, eval time: {eval_time:.4f}')
        if config.mode != 'eval':
            if beam_size == 3 and (epoch % 5) == 0:
                for metric, values in beam_metrics.items():
                    main_logger.info(f'beam search (size 3): {metric:<7s}: {values["score"]:7.4f}')
            spiders.append(spider)
            if spider >= max(spiders):
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "beam_size": beam_size,
                            "epoch": epoch},
                            str(model_output_dir) + '/best_model.pt'.format(epoch))
        else:
            eval_spiders.append(spider)
            if spider >= max(eval_spiders):
                eval_metrics.update(beam_metrics)


def test_beam(beam_size, max_len=30):

    model.eval()
    with torch.no_grad():
        with open('test_output.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['file_name', 'caption_predicted'])
            for src, file_names in tqdm(test_data, total=len(test_data)):
                src = src.to(device)
                output = beam_search(model, src, sos_ind=sos_ind, eos_ind=eos_ind, beam_size=beam_size)

                output_batch = []
                for sample in output:
                    sample = sample[1:]
                    sample_words_ind = []
                    for sample_word in sample:
                        if sample_word == eos_ind:
                            break
                        sample_words_ind.append(sample_word)
                    caption_word = [words_list[index] for index in sample_words_ind]
                    caption_str = ' '.join(caption_word)
                    output_batch.append(caption_str)
                for caption, file_name, in zip(output_batch, file_names):
                    writer.writerow([file_name, caption])

def extract_features(data_split: torch.utils.data.DataLoader):
    model.eval()
    with torch.no_grad():
        for batch_idx, eval_batch in tqdm(enumerate(data_split), total=len(data_split)):
            src, tgt, f_names, tgt_len, captions = eval_batch
            src = src.to(device)
            import pdb; pdb.set_trace()
            encoder_output = model.feature_extractor(src)


def get_fc1(self):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print(self.x_fc1)

parser = argparse.ArgumentParser(description='Settings for audio caption model')

parser.add_argument('-n', '--exp_name', type=str, default='exp_feats', help='name of the experiment')
parser.add_argument('-m', '--mask_type', type=str, default='zero_value', help='masking type')

args = parser.parse_args()

config = get_config()

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

dataset = config.data.type

batch_size = config.data.batch_size
num_workers = config.data.num_workers
input_field_name = config.data.input_field_name

# data loading
if dataset == 'clotho':
    words_list_path = config.path.clotho.words_list
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
    words_list_path = config.path.audiocaps.words_list
    # words_freq_path = config.path.audiocaps.words_freq
    training_data = get_audiocaps_loader(split='train',
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=num_workers)

    validation_data = get_audiocaps_loader(split='val',
                                           batch_size=batch_size,
                                           num_workers=num_workers)

    evaluation_data = get_audiocaps_loader(split='test',
                                           batch_size=batch_size,
                                           num_workers=num_workers)

# loading vocabulary list
if config.mode == 'finetune' and config.finetune.audiocap:
    words_list_path = 'data/pickles/new_words_list.p'
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

model.register_forward_hook(get_fc1)
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
    extract_features(data_split)

main_logger.info('Evaluation done.')
for metric, values in eval_metrics.items():
    main_logger.info(f'best metric: {metric:<7s}: {values["score"]:7.4f}')

