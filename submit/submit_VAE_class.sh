#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N VAE_class
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/VAE_class_1.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/VAE_class_1.log
#PBS -l nodes=1:ppn=10
#PBS -l mem=45gb
#PBS -l walltime=02:00:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/TEST/VAE_classification.py --num_bins 10000 --BERT_tokens 0.0125 --case no_CLS_no_desc_no_rettime --class_param Group2 --kleiner_type --latent_dim 2 --n_linear_layers 4 --n_units 64 --n_epochs 20 --sample_size 16
