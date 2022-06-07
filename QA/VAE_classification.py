#!/usr/bin/env python

import torch
from torch import nn
import argparse 

from data_load import *
from utils import *
from architectures import *


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define vocabulary (special tokens already considered) and type if classification layer
    vocab = get_vocab(num_bins=args.num_bins)

    class_param_long = 'kleiner' + args.kleiner_type if args.kleiner_type else args.class_param 

    # Define pathways for the different uploads and savings
    files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/' + 'BERT_tokens_' + args.BERT_tokens + '/' + args.case +  '_' + str(args.num_bins) + '_tokens/'
    evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/VAE/Classification/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) + '.txt', "w")

    # Upload the trained VAE encoder weigths
    weights_pathway = '/home/projects/cpr_10006/people/enrcop/models/VAE/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens + '_' + str(args.num_bins) + '.pt'

    # Pathways to save figures
    save_loss = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Classification/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens + '_' + str(args.num_bins) + '_loss.png'
    ROC_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Classification/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) +  '_ROC.png'
    cm_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Classification/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens + '_' + str(args.num_bins) +'_cm.png'

    # Define the datasets 
    train_ds, val_ds, num_labels, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                    vocab, 
                                                                    training_percentage=0.7, 
                                                                    validation_percentage=0.2, 
                                                                    crop_input = True, 
                                                                    class_param = args.class_param,
                                                                    kleiner_type=args.kleiner_type)

    # Define the models and upload the weigths for the VAE
    VAE_model = ConvVAE_encoder(n_channels=args.num_channels,
                    kernel=args.kernel,
                    stride=args.stride,
                    padding=1,
                    latent_dim=args.latent_dim).to(device)

    checkpoint = torch.load(weights_pathway, map_location=device)
    VAE_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

    #Define hyperparameters
    epochs = args.n_epochs
    sample_size = args.sample_size
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    training_loss = []; validation_loss = []; accuracy_evolution = []

    for lr in args.lr:
        
        evolution_file.write(f'Learning rate: {lr:e} \n')
        
        classification_network = LinearClassificationLayer(n_input_features=args.latent_dim, num_labels=num_labels, n_layers=args.n_linear_layers, n_units=args.n_units)

        optimizer = torch.optim.Adam(classification_network.parameters(), lr = lr)
    
        training_loss_lr = []; validation_loss_lr = []
        for epoch in range(1, epochs+1):
            
            train_loss = train_class_VAE(VAE_model, classification_network, vocab, train_ds, device, optimizer, criterion, batchsize=sample_size)
            training_loss_lr.append(train_loss)

            val_loss, accuracy, y_true, y_pred = val_class_VAE(VAE_model, classification_network, vocab, val_ds, device, optimizer, criterion, batchsize=sample_size)
            validation_loss_lr.append(val_loss)
            accuracy_evolution.append(accuracy)

            if val_loss < best_val_loss:
                best_lr = lr
                best_val_loss = val_loss
                best_accuracy = accuracy
                best_y_pred = y_pred
                best_y_true = y_true
            
            evolution_file.write(f'[Epoch {epoch:3d}]: Training loss= {train_loss:.4f} | Validation loss= {val_loss:.4f} | Accuracy= {accuracy:.4f} \n')
        #evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
        #for i, accuracy_epoch in enumerate(accuracy_evolution):
        #    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}] ')
        #evolution_file.write('\n')
        training_loss.append(training_loss_lr)
        validation_loss.append(validation_loss_lr)
        
    confusion_matrix, precision, recall, F1_score = get_metrics(best_y_true, best_y_pred, cm_pathway, ROC_curve=True, ROC_fig_pathway=ROC_pathway)
    plt.clf()
    plot_loss(epochs, args.lr, training_loss, validation_loss, pathway=save_loss)

    evolution_file.write(f'Accuracy for model with lower validation loss: {best_accuracy:3.2f}, with learning rate: {best_lr:e} \n')
    evolution_file.write(f'Precision {precision:.3f} | Recall {recall:.3f} | F1_score {F1_score:.3f}')
    evolution_file.close()

if __name__ == '__main__':
    
    # create the parser
    parser = argparse.ArgumentParser(prog='VAE_classification.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), 
        description='''Use trained VAE encoder for classification from latent space.''')
    
    parser.add_argument('--num_bins', help='Number of bins for the creation of the vocabulary', required=True, type=int)
    parser.add_argument('--BERT_tokens', help='Directory where to find the BERT tokens', required=True)
    parser.add_argument('--case', help='Combination of parameters for tokens used', required=True)
    parser.add_argument('--class_param', help='classification parameter (Group2/HPvsALD, nas_inflam, kleiner, nas_steatosis_ordinal', required=True)
    parser.add_argument('--kleiner_type', help='Type of kleiner when class_param is kleiner (significant or advanced')
    parser.add_argument('--num_channels', help='Number channels for 2D CNN',default=2, type=int)
    parser.add_argument('--kernel', help='Kernel size for 2D CNN',default=5, type=int)
    parser.add_argument('--stride', help='Stride for the 2D CNN', default=2, type=int)
    parser.add_argument('--latent_dim', help='Latent dimension used for the latent space', default=2, type=int)
    parser.add_argument('--n_linear_layers', help='Number linear layers for the linear classification network', default=4, type=int)
    parser.add_argument('--n_units', help='Number of units for the inner linear layers for classification network', default=32, type=int)
    parser.add_argument('--lr', action='append', help='Learning rate', required='True', type=float)
    parser.add_argument('--n_epochs', help='number epochs', default=30, type=int)
    parser.add_argument('--sample_size', help='Equivalent to batch_size, but each sample is an input in this case', default=32, type=int)
    
    args = parser.parse_args()

    main(args)


