# Mass_spec_NLP_MSc_thesis
This is a repository used during the development of the Master's thesis for MSc Bioinformatics and Systems Biology DTU 2022 with title "Disease prediction from plasma proteomics using Natural Language Processing".

Packages required:
- Pytorch (Using preferred command from https://pytorch.org/get-started/locally/ for your specific case)
- torch.text (https://pypi.org/project/torchtext/)
- pandas
- umap-learn (pip install umap-learn)


The natural process to follow is as shown below:
1. dataset_translator.py 
2. BERT_train.py

If one wants to analyze the performance of the trained BERT model before going into the fine-tuning phase, the BERT train analysis branch includes:

    3. BERT_train_analysis.py
    4. BERT_scan_detection.py
    5. plot_samples.py

Otherwise, for the fine-tuning step:
6. BERT_finetune.py

The other support scripts:
7. architectures.py
8. utils.py
9. data_load.py

,contain model architectures and support functions required through the whole process on the other scripts.

![My Image](Images/Workflow_2.png)
