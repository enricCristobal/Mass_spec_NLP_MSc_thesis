#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N BERT_tokens8
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/tokens.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/tokens.log
#PBS -l nodes=1:ppn=40
#PBS -l mem=190gb
#PBS -l walltime=30:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/QA/dataset_translator.py --num_bins 50000 --peak_filter 0.1 --case no_CLS_desc_rettime --input_size 512 --CLS_token "" --desc_order True  --ret_time True
