#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N VAE_train
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/VAE.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/VAE.log
#PBS -l nodes=1:ppn=20
#PBS -l mem=94gb
#PBS -l walltime=6:00:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/QA/VAE.py --num_bins 50000 --BERT_tokens 0.1 --case no_CLS_desc_no_rettime --class_param kleiner --kleiner_type significant --labels Significant_fibrosis --labels No_fibrosis --n_channels 4 --kernel 5 --stride 2 --latent_dim 2 --lr 5e-5 --lr 1e-5 --n_epochs 30 --batchsize 32 
