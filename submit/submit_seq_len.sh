#!/bin/sh
#PBS -W group_list=cpr_10006 -A cpr_10006
#PBS -N seq_plot
#PBS -e /home/projects/cpr_10006/people/enrcop/code_logs_errors/seq_plot.err
#PBS -o /home/projects/cpr_10006/people/enrcop/code_logs_errors/seq_plot.log
#PBS -l nodes=1:ppn=10
#PBS -l mem=45gb
#PBS -l walltime=10:00

module load tools
module load anaconda3/2021.11

source ~/.bashrc
conda init bash
conda activate myenv

python /home/projects/cpr_10006/people/enrcop/TEST/seq_len_count.py 1 True

