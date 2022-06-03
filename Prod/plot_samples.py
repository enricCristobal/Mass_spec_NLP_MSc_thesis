#!/usr/bin/env python
from sre_constants import CATEGORY_DIGIT
import sys

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
data_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\test_data\\' 
labels_dir = data_dir + 'ALD_histological_scores.csv'

embedding_analysis_dir = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Test\\embedding_analysis_data\\'

# Get the samples used for the ploting
samples, _ = divide_train_val_samples(embedding_analysis_dir, 1, 0) 
# !!LOCAL THING: when plotting applying the mean to all the scans per one sample, 0.9 to avoid histological scores csv file whn using test_data dir
# when not applying any aggregation and obtaining all scans per sample, use 1 and place just one file in embedding_analysis_dir
dataset, _, _ = FineTuneBERTDataLoader(embedding_analysis_dir, vocab, training_percentage=1, validation_percentage=0,\
    class_param=sys.argv[1], kleiner_type = sys.argv[2], labels_path=labels_dir)

# Define the model network for BERT model and load trained weights
BERT_model = BERT_trained(ntoken = len(vocab), 
            d_model = 192, 
            d_hid = 192,
            nhead = 3,  
            nlayers = 3, 
            activation = F.gelu, 
            dropout = 0.1).to(device)

# If decoder layer removal for fine-tuning done and weights saved:
#model_weights = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/BERT_vanilla_small/bert_vanilla_small_weights.pt'
#BERT_model.load_state_dict(torch.load(model_weights, map_location=device))
##LOCAL PATHWAY
model_weights = 'C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\models\\bert_vanilla_small_weights.pt'
BERT_model.load_state_dict(torch.load(model_weights, map_location=device))

#checkpoint = torch.load(model_weights, map_location=device)
#BERT_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

plot_BERT_embeddings(model=BERT_model,
                samples_names=samples,
                dataset=dataset,
                batchsize = 32,
                aggregation_layer=None, # For now just 'sample' and 'ret_time'
                labels = [sys.argv[3], sys.argv[4]], #['Healthy', 'ALD'],
                pathway='C:\\Users\\enric\\OneDrive\\Escriptori\\TFM\\01_Code\\Code\\Computerome_results\\BERT_vanilla_embeddings\\Individual_analysis\\',
                #pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/BERT_vanilla/' + sys.argv[5],  #Group2_HPvsALD/Embedding1.0.png',
                patient_type= 'HP_2',
                device = device)

