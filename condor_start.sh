# python finetune_rl.py -n dropmodel -lr 0.00001 -warmup yes -model_path model_ckpt/drop_model.pt -batch 16
eval "$('/vol/research/Sentiment/qh00006/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate DCASE2021_task6_v2
cd /vol/research/AAC_CVSSP_research/yuanjing/DCASE2021_task6_v2/
python finetune_rl.py -n $1 \
 -lr $2 \
 -warmup $3 \
 -model_path $4 \
 -batch 32

