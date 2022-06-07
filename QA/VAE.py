#!/usr/bin/env python

import torch
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
    files_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/' + 'BERT_tokens_' + args.BERT_tokens + '/' + args.case +  '_' + str(args.num_bins) + '_tokens/' #BERT_tokens_0.0125/no_CLS_no_desc_no_rettime_10000_tokens/'
    evolution_file = open('/home/projects/cpr_10006/people/enrcop/loss_files/VAE/Train/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) + '.txt', "w") #group2_no_cls_no_desc_no_rettime_0.0125_10000.txt', "w")
    
    # Save model and loss figures
    save_model = '/home/projects/cpr_10006/people/enrcop/models/VAE/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) + '.pt' #group2_no_CLS_no_desc_no_rettime_10000.pt'
    save_loss = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/Train/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) + '.png' #_group2_no_CLS_no_desc_no_rettime_10000.png'
    save_latent_fig = '/home/projects/cpr_10006/people/enrcop/Figures/VAE/latent_space/' + class_param_long + '/' + args.case +  '_' + args.BERT_tokens + '_' + str(args.num_bins) + '_latent.png' #group2_no_CLS_no_desc_no_rettime_10000.png'

    # Define dataset
    train_ds, val_ds, _, min_scan_count = FineTuneBERTDataLoader(files_dir, 
                                                                    vocab, 
                                                                    training_percentage=0.7, 
                                                                    validation_percentage=0.2, 
                                                                    crop_input = True, 
                                                                    class_param = args.class_param,
                                                                    kleiner_type= args.kleiner_type)
    
    # Define training parameters and criterion
    epochs = args.n_epochs
    batch_size = args.batchsize
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model = None
    training_loss = []; validation_loss = []

    for lr in args.lr:
        
        evolution_file.write(f'Learning rate: {lr:e} \n')

        # Define model
        model = ConvVAE(n_channels=args.n_channels,
                        kernel=args.kernel,
                        stride=args.stride,
                        padding=1,
                        input_heigth=min_scan_count,
                        input_width=512,
                        latent_dim=args.latent_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
        training_loss_lr = []; validation_loss_lr = []

        for epoch in range(1, epochs+1):
         
            train_loss, latent_space_samples, labels = train_VAE(model, vocab, train_ds, device, optimizer, criterion, batchsize=batch_size)
            training_loss_lr.append(train_loss)

            val_loss = val_VAE(model, vocab, val_ds, device, optimizer, criterion, batchsize=batch_size)
            validation_loss_lr.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_latent_space = latent_space_samples
                

            evolution_file.write(f'[Epoch {epoch:3d}]: Training loss: {train_loss:.4f} / Validation loss: {val_loss:.4f}  \n')

        training_loss.append(training_loss_lr)
        validation_loss.append(validation_loss_lr)

    # Save the best model
    torch.save({'epoch': epoch,
                        'learning_rate': lr,
                        'model_state_dict': best_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss
                            }, save_model)

    # Plot the different figures
    plot_loss(epochs, args.lr, training_loss, validation_loss, save_loss)
    plt.clf()
    plot_VAE_embedding(best_latent_space, labels, args.labels, save_latent_fig)


if __name__ == '__main__':
    
    # create the parser
    parser = argparse.ArgumentParser(prog='VAE.py', formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=50, width=130),
        description='Train encoder and decoder VAE based on reconstruction error')
    
    parser.add_argument('--num_bins', help='Number of bins for the creation of the vocabulary', required=True, type=int)
    parser.add_argument('--BERT_tokens', help='Directory where to find the BERT tokens', required=True)
    parser.add_argument('--case', help='Combination of parameters for tokens used', required=True)
    parser.add_argument('--class_param', help='classification parameter (Group2/HPvsALD, nas_inflam, kleiner, nas_steatosis_ordinal', required=True)
    parser.add_argument('--labels', action='append', help='Names for the different labels for when plotting the latent space', required=True)
    parser.add_argument('--kleiner_type', help='Type of kleiner when class_param is kleiner (significant or advanced')
    parser.add_argument('--n_channels', help='Base for number channels used in Conv2d, then inside the model these are multiplied by 2 and 4 for different layers', default=4, type=int)
    parser.add_argument('--kernel', help='Kernel for the Conv2d in the different layers of the VAE', default=5, type=int)
    parser.add_argument('--stride', help='Stride for the Conv2d in the different layers of the VAE', default=2, type=int)
    parser.add_argument('--latent_dim', help='Latent dimension used for the latent space', default=2, type=int)
    parser.add_argument('--lr', action='append', help='Different learning rates used for the VAE', required=True, type=float)
    parser.add_argument('--n_epochs', help='number epochs', default=30, type=int)
    parser.add_argument('--batchsize', default=32, type=int)

    args = parser.parse_args()

    main(args)
