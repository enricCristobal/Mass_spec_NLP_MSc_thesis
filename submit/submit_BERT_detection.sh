#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N BERT_detection
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/BERT_detection.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/BERT_detection.log
#PBS -l nodes=1:ppn=39
#PBS -l mem=180gb
#PBS -l walltime=3:00:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/PROD/BERT_scan_detection.py --num_bins 10000 --BERT_tokens 0.1 --BERT_type BERT_small --case no_CLS_desc_no_rettime --data_repetition 15 --batchsize 32 
