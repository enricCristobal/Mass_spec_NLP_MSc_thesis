#!/usr/bin/env python
from utils_clean import *
from architectures_clean import *
from data_load_clean import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered)
vocab = get_vocab(num_bins=50000)

# Define directory where sample files are found
#data_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'
# !!labels_path define correctly default DataLoader function in data_load_clean
##LOCAL PATHWAY
data_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\' 
labels_dir = data_dir + 'ALD_histological_scores.csv'

# Get the samples used for the ploting
samples, _ = divide_train_val_samples(data_dir, 0.9, 0) # !!LOCAL THING: 0.9 to avoid histological scores csv file

dataset, _, _ = FineTuneBERTDataLoader(data_dir, vocab, training_percentage=0.9, validation_percentage=0,\
    class_param='kleiner', kleiner_type = 'advanced', labels_path=labels_dir)

# Define the model network for BERT model and load trained weights
BERT_model = BERT_trained(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

# If decoder layer removal for fine-tuning done and weights saved:
#model_weights = '/home/projects/cpr_10006/people/enrcop/models/bert_vanilla_small_weights.pt'
#BERT_model.load_state_dict(torch.load(model_weights, map_location=device))
##LOCAL PATHWAY
model_weights = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\bert_vanilla_small_weights.pt'
BERT_model.load_state_dict(torch.load(model_weights, map_location=device))

'''
# When saved properly with the General Checkpoint and decoder layer removal for fine tuning not done yet
BERT_model = BERT_trained(ntokens, emsize, nhead, d_hid, nlayers, activation, dropout).to(device)
optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-4)

checkpoint = torch.load('/home/projects/cpr_10006/people/enrcop/models/{model_name}.pt')
BERT_model.load_state_dict(checkpoint['model_state_dict'])

# Check this part comparing to BERT_fine_tune_test.py process
state_dict = copy.deepcopy(BERT_model.state_dict())
del state_dict['decoder.weight']
del state_dict['decoder.bias']
'''

plot_embeddings(model=BERT_model,
                samples_names=samples,
                dataset=dataset,
                batchsize = 32,
                aggregation_layer='sample', # For now just 'sample' and 'ret_time'
                pathway='C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Dummy_results\\Embedding2.png',
                labels = ['Healthy', 'ALD'],
                #pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_vanilla/Group2_HPvsALD/Embedding1.0.png',
                device = device)

