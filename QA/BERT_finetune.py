#!/usr/bin/env python

import torch
from torch import nn
import time
import argparse

from data_load import *
from utils import *
from architectures import *

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(52)
    torch.cuda.manual_seed(52)

    # Define vocabulary (special tokens already considered) and type if classification layer
    vocab = get_vocab(num_bins=10000)

    class_param_long = 'kleiner' + args.kleiner_type if args.kleiner_type else args.class_param 

    # Define the architecture for the classification layer for the fine-tuning: crop_input required if CNN
    class_layer = args.class_layer # 'Linear' or 'CNN'
    att_matrix = args.att_matrix #False # If CNN att_matrix must always be
    crop_input = True if att_matrix else False
    print(crop_input)

    # Define pathways for the different uploads and savings
    files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/' + 'BERT_tokens_' + args.BERT_tokens + '/' + args.case +  '_' + str(args.num_bins) +  '_tokens/'
    evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/BERT_finetune/' + args.BERT_type + '/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) + '_' + str(args.class_layer) + '_' + str(args.att_matrix) + '.txt', "w")

    # Load BERT weights
    model_weights = '/home/projects/cpr_10006/people/enrcop/models/BERT_train/' + args.BERT_type + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '.pt'

    # Save model weights
    save_model = '/home/projects/cpr_10006/people/enrcop/models/BERT_finetune/' + args.BERT_type + '/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.class_layer) + '_' + str(args.att_matrix) + '.pt'

    # Figures pathways
    error_plot_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/' + args.BERT_type + '/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.class_layer) + '_' + str(args.att_matrix) + '_loss.png'
    att_weights_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/' + args.BERT_type + '/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.class_layer) + '_' + str(args.att_matrix) + '_att_weigths.png'
    ROC_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/' + args.BERT_type + '/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.class_layer) + '_' + str(args.att_matrix)+ '_ROC_curve.png'
    cm_pathway = '/home/projects/cpr_10006/people/enrcop/Figures/BERT_finetune/' + args.BERT_type + '/' + class_param_long + '/' + args.case + '_' + args.BERT_tokens +  '_' + str(args.num_bins) + '_' + str(args.class_layer) + '_' + str(args.att_matrix) + 'conf_matrix.png'

    # Save fine-tune model
    #save_finetuning_model = '/home/projects/cpr_10006/people/enrcop/models/BERT_finetune/BERT_small/bert_Group2_HPvsALD.pt'

    train_finetune_ds, val_finetune_ds, num_labels, _ = FineTuneBERTDataLoader(files_dir, 
                                                                    vocab, 
                                                                    training_percentage=0.3, 
                                                                    validation_percentage=0.15, 
                                                                    crop_input=crop_input, 
                                                                    class_param = args.class_param,
                                                                    kleiner_type=args.kleiner_type)

    # Create the model network
    # When saved properly with the General Checkpoint and decoder layer removal for fine tuning not done yet
    # Define the model network for BERT model and load trained weights

    if args.BERT_type == 'BERT_small':
        d_model = 192; d_hid = 192; nhead = 3; nlayers = 3
    elif args.BERT_type ==  'BERT_half':
        d_model = 384; d_hid = 384; nhead = 6; nlayers = 6

    # Define the model network for training BERT model
    BERT_model = BERT_trained(ntoken=len(vocab),  # size of vocabulary 
                            d_model=d_model, # embedding dimension
                            d_hid=d_hid, # dimension of the feedforward network model in nn.TransformerEncoder
                            nhead=nhead, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                            nlayers=nlayers,# number of heads in nn.MultiheadAttention
                            activation=F.gelu, # activation function on the intermediate layer
                            dropout=0.1).to(device) # dropout probability


    checkpoint = torch.load(model_weights, map_location=device)
    BERT_model.load_state_dict(checkpoint['model_state_dict'], strict = False)

    # Define fine-tuning hyperparameters
    epochs = args.n_epochs
    batchsize = args.batchsize
    criterion = nn.CrossEntropyLoss()
    #scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')
    best_model = None
    training_loss = []; validation_loss = []; accuracy_evolution = []

    for lr in args.lr:
        print('Learning rate: ', lr)
        evolution_file.write(f'Learning rate: {lr:e} \n')

        # This way BERT weights are frozen through the finetuning training
        # Define classification network architecture and initialize model for each learning rate
        model, BERT_finetune_train_fun = define_architecture(class_layer=class_layer,
                                                att_matrix=att_matrix,
                                                n_input=d_model,
                                                num_labels=num_labels,
                                                n_layers_attention=args.n_layers_attention,
                                                n_units_attention=args.n_units_attention,
                                                num_channels=args.num_channels,
                                                kernel_size=args.kernel_size,
                                                padding=1,
                                                n_units_linear_CNN=args.n_units_linear_CNN,
                                                n_layers_linear=args.n_layers_linear,
                                                n_units_linear=args.n_units_linear)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        training_loss_lr = []; validation_loss_lr = []; accuracy_evolution = []

        for epoch in range(1, epochs + 1):
            print('Epoch: ', epoch, '/', epochs)
            epoch_start_time = time.time()
            
            train_error, att_weights_matrix = BERT_finetune_train_fun(BERT_model, model, optimizer, criterion, learning_rate=lr, 
            dataset=train_finetune_ds, results_file=evolution_file, batchsize=batchsize, epoch=epoch, write_interval = args.write_interval, device=device,
            scaler=None)

            training_loss_lr.append(train_error)

            val_loss, acc, y_true,y_pred = BERT_finetune_evaluate(BERT_model, model, att_matrix=att_matrix, class_layer=class_layer, criterion=criterion, 
            dataset=val_finetune_ds, results_file=evolution_file, batchsize=batchsize, current_epoch=epoch, start_time=epoch_start_time, device=device)
            
            validation_loss_lr.append(val_loss)
            accuracy_evolution.append(acc)

            if val_loss < best_val_loss:
                best_lr = lr
                best_model = model
                best_val_loss = val_loss
                best_accuracy = acc
                best_y_pred = y_pred
                best_y_true = y_true
                if att_matrix:
                    best_att_weights_matrix = update_best_att_matrix(att_weights_matrix)
        
        training_loss.append(training_loss_lr)
        validation_loss.append(validation_loss_lr)

    # Save the best model for future use
    #torch.save(best_model.state_dict(), save_model)

    # Get the different metrics for the best model and print all the plots obtained
    confusion_matrix, precision, recall, F1_score = get_metrics(best_y_true, best_y_pred, cm_pathway, ROC_curve=True, ROC_fig_pathway=ROC_pathway)
    
    plot_loss(epochs, args.lr, training_loss, validation_loss, pathway=error_plot_pathway)


    if att_matrix:
        plot_att_weights(best_att_weights_matrix, pathway=att_weights_pathway)

    #evolution_file.write('Accuracy evolution per epoch: [epoch, accuracy] \n')
    #for i, accuracy_epoch in enumerate(accuracy_evolution):
    #    evolution_file.write(f'  [Epoch {i:3d}:  {accuracy_epoch:3.2f}]  ')
    #evolution_file.write('\n')

    # Write the metrics for the best model at the end of the evolution file
    evolution_file.write(f'Accuracy for model with lower validation loss: {best_accuracy:3.2f}, with learning rate: {best_lr:e} \n')
    evolution_file.write(f'Precision {precision:.3f} | Recall {recall:.3f} | F1_score {F1_score:.3f}')
    evolution_file.close()


if __name__ == '__main__':
    
    # create the parser
    parser = argparse.ArgumentParser(prog='BERT_finetune.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130), 
        description="Finetune BERT by training a classification network (either linear or convolutional). ")
    
    parser.add_argument('--num_bins', help='Number of bins for the creation of the vocabulary', required=True, type=int)
    parser.add_argument('--BERT_tokens', help='Directory where to find the BERT tokens', required=True)
    parser.add_argument('--BERT_type', help='Type/Size of the BERT model to use (small, half)', required=True)
    parser.add_argument('--case', help='Combination of parameters for tokens used', required=True)
    parser.add_argument('--class_layer', help='Type of classification layer used (Linear or CNN)', required=True)
    parser.add_argument('--att_matrix', help='Boolean to decide if we want attention matrix', type=bool, required=True)
    parser.add_argument('--class_param', help='Classification parameter (Group2/HPvsALD, nas_inflam, kleiner, nas_steatosis_ordinal', required=True)
    parser.add_argument('--kleiner_type', help='Type of kleiner when class_param is kleiner (significant or advanced')
    parser.add_argument('--n_epochs', help='number epochs', default=40, type=int)
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', action='append', help='Different learning rates used for the classification of layer after BERT', required=True, type=float)
    parser.add_argument('--n_layers_attention', help='Number layers linear network for attention', type=int)
    parser.add_argument('--n_units_attention', help='Number neurons for each linear layer for attention', type=int)
    parser.add_argument('--num_channels', help='Number channels for 2D CNN (when applied)', type=int)
    parser.add_argument('--kernel_size', help='Kernel size for 2D CNN (when applied)', type=int)
    parser.add_argument('--n_units_linear_CNN', help='Number neurons for linear network after CNN to get 1 prediction per sample for the label (for now always binary classification)', type=int)
    parser.add_argument('--n_layers_linear', help='Number layers linear network for classification (when applied)', type=int)
    parser.add_argument('--n_units_linear', help='Number neurons for each linear layer for classification (if applied)', type=int)
    parser.add_argument('--write_interval', help='Interval to write in the loss file (either # samples or # batches depending on class layer)', type=int, required=True)
    
    args = parser.parse_args()


    main(args)
