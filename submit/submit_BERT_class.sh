#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N BERT_finetune_1
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/BERT_fientune_1.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/BERT_finetune_1.log
#PBS -l nodes=1:ppn=20:gpus=1
#PBS -l mem=96gb
#PBS -l walltime=12:00:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/QA/BERT_finetune.py --num_bins 10000 --BERT_tokens 0.0125 --BERT_type BERT_small --case no_CLS_no_desc_no_rettime --class_layer CNN --att_matrix True --class_param Group2 --kleiner_type None --n_epochs 5 --batchsize 16 --lr 3e-5 --lr 4e-4 --lr 5e-5 --n_layers_attention 3 --n_units_attention 64 --num_channels 4 --kernel_size 5 --n_units_linear_CNN 128 --n_layers_linear None --n_units_linear None --write_interval 50 
