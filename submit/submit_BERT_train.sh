#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N BERT_half_big
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/BERT_half_big.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/BERT_half_big.log
#PBS -l nodes=1:ppn=39:gpus=1
#PBS -l mem=180gb
#PBS -l walltime=1:00:00:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/QA/BERT_train.py --num_bins 50000 --BERT_tokens 0.1 --BERT_type BERT_half --case no_CLS_desc_no_rettime --CLS_token "" --add_ret_time "" --input_size 512 --data_repetition 1 --n_epochs 2 --batchsize 32 --limited_seq_len 128 --perc_epochs_shorter 0.4

