#!/usr/bin/env python
from sre_constants import CATEGORY_DIGIT
import sys
import copy

from utils import *
from architectures import *
from data_load import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define vocabulary (special tokens already considered)
num_bins=50000
vocab = get_vocab(num_bins)

# Define directory where sample files are found
#data_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/BERT_tokens/scan_desc_tokens_CLS/'
# !!labels_path define correctly default DataLoader function in data_load_clean
##LOCAL PATHWAY
#data_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\tokens_0.0125_no_CLS_no_desc_no_rettime_10000\\' 
labels_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\ALD_histological_scores.csv'

embedding_analysis_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\dummy_data\\'

# Get the samples used for the ploting
samples, _ = divide_train_val_samples(embedding_analysis_dir, 1, 0) 
# !!LOCAL THING: when plotting applying the mean to all the scans per one sample, 0.9 to avoid histological scores csv file whn using test_data dir
# when not applying any aggregation and obtaining all scans per sample, use 1 and place just one file in embedding_analysis_dir
dataset, _, _ = FineTuneBERTDataLoader(embedding_analysis_dir, vocab, training_percentage=1, validation_percentage=0,\
    class_param=sys.argv[1], kleiner_type = sys.argv[2], labels_path=labels_dir)

'''
# When saved properly with the General Checkpoint and decoder layer removal for fine tuning not done yet
# Define the model network for BERT model and load trained weights
BERT_model = TransformerModel(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

optimizer = optimizer = torch.optim.AdamW(BERT_model.parameters(), lr=1e-4, eps=1e-4)

#checkpoint = torch.load('/home/projects/cpr_10006/people/enrcop/models/{model_name}.pt')
checkpoint = torch.load('C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\BERT_small_no_CLS_no_desc_no_rettime_0.0125_10000.pt', 
map_location=device)
BERT_model.load_state_dict(checkpoint['model_state_dict'])

# Check this part comparing to BERT_fine_tune_test.py process
state_dict = copy.deepcopy(BERT_model.state_dict())
del state_dict['decoder.weight']
del state_dict['decoder.bias']
'''
model = BERT_trained(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

#model.load_state_dict(state_dict)

# If decoder layer removal for fine-tuning done and weights saved:
##LOCAL PATHWAY
model_weights = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\bert_vanilla_small_weights.pt'
model.load_state_dict(torch.load(model_weights, map_location=device))


plot_BERT_embeddings(model=model,
                samples_names=samples,
                dataset=dataset,
                batchsize = 32,
                aggregation_layer='sample', # For now just 'sample' and 'ret_time'
                labels = [sys.argv[3], sys.argv[4]], #['Healthy', 'ALD'],
                pathway = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\',
                #pathway='C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Computerome_results\\BERT_vanilla_embeddings\\Individual_analysis\\' + sys.argv[5],
                #pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_vanilla/' + sys.argv[5],  #Group2_HPvsALD/Embedding1.0.png',
                patient_type= 'HP_2',
                device = device)

