#!/usr/bin/env python
from functions import *
from architectures import *
from data_load import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered)
vocab = get_vocab(num_bins=50000)

# Define directory where sample files are found and where we want to save the plot
data_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'
plot_dir = '/home/projects/cpr_10006/people/enrcop/Figures/Embeddings/BERT_vanilla_HPvsALD.png'

## !!TODO: Put all these four commands in one simple function to get everything done in one line
# Get the samples used for the ploting
samples, _ = divide_train_val_samples(data_dir, 0.4, 0.1) # !!LOCAL THING: 0.9 to avoid histological scores csv file

# Get a dataframe where we obtain the label for each sample (healthy vs. disease, or different levels of disease, etc.)
labels_df = fine_tune_ds(samples, class_param = 'Group2')
# From labels_df we can get which label corresponds to each class used for classification, i.e. 1 is ALD and 0 is HP
 
#Get labels for chosen samples in a list and get rid of potential "QC"/controls of our chosen list of samples
labels, samples = get_labels(df=labels_df, samples_list=samples)

# Create the dataset that will go through the model and will be used for plotting
dataset = [create_finetuning_dataset(f, vocab, data_dir, labels[i]) for i,f in enumerate(samples)]

# Define the model network for BERT model and load trained weights
BERT_model = BERT_trained(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

# If decoder layer removal for fine-tuning done and weights saved:
BERT_model.load_state_dict(torch.load('/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small_weights.pt', \
                            map_location=device))

plot_embeddings(model=BERT_model,
                samples_names=samples,
                dataset=dataset,
                batchsize = 32,
                aggregation_layer='sample', # For now just 'sample' and 'ret_time'
                plot_dir = plot_dir,
                device = device)

