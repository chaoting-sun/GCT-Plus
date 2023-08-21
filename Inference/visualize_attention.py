import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from bertviz import model_view
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from Model.build_model import get_sampler
from Utils.smiles import murcko_scaffold


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffoldSmiles(mol=mol)


def plot_self_attention(attn, tokens, save_path):
    plt.figure()
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add colorbar to show the scale
    plt.xlabel("SMILES")
    plt.ylabel("SMILES")

    plt.xticks(range(len(tokens)), tokens)
    plt.yticks(range(len(tokens)), tokens)
    
    plt.savefig(save_path)


def plot_cross_attention(attn, in_tokens, out_tokens, save_path):
    plt.figure()
    plt.imshow(attn, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Add colorbar to show the scale
    plt.xlabel("SMILES")
    plt.ylabel("SMILES")

    plt.xticks(range(len(in_tokens)), in_tokens)
    plt.yticks(range(len(out_tokens)), out_tokens)
    
    plt.savefig(save_path)


def plot_attention_map(attention_matrix, tokens, save_path, h=8):
    num_rows = 2
    num_cols = 4
    assert len(attention_matrix) == num_cols * num_rows
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 12))
    axs = axs.flatten()

    for head in range(h):
        # Get the attention matrix for the current head
        attention_head = attention_matrix[head]
        
        print(attention_head.shape)

        # Plot the attention map for the current head
        axs[head].imshow(attention_head, cmap='hot')
        axs[head].set_title(f'Head {head+1}')
        
        # Set the x-axis and y-axis ticks
        axs[head].set_xticks(range(len(tokens)))
        axs[head].set_yticks(range(len(tokens)))
        axs[head].set_xticklabels(tokens)
        axs[head].set_yticklabels(tokens)
        
        # Add colorbar
        # cbar = fig.colorbar(axs[head].imshow(attention_head, cmap='hot'), ax=axs[head])
        # cbar = axs[head].collections[0].colorbar
        # cbar.set_label('Attention Score')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(save_path)


def plot_attention_line(attention_matrix, tokens, save_path, h=8):
    num_rows = 2
    num_cols = 4
    assert len(attention_matrix) == num_cols * num_rows
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 12))
    axs = axs.flatten()  # Flatten the axs list
    
    for head in range(h):
        attention_head = attention_matrix[head]
        
        # Plot the attention map for the current head
        axs[head].imshow(attention_head, cmap='hot')
        axs[head].set_title(f'Head {head+1}')
        
        # Set the x-axis and y-axis ticks
        axs[head].set_xticks(range(len(tokens)))
        axs[head].set_yticks(range(len(tokens)))
        axs[head].set_xticklabels(tokens)
        axs[head].set_yticklabels(tokens)
        
        # Connect the right and left tokens
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                axs[head].plot([i, j], [i, j], color='blue', alpha=attention_head[i, j])
        
        # Add colorbar
        cbar = fig.colorbar(axs[head].imshow(attention_head, cmap='hot'), ax=axs[head])
        cbar.set_label('Attention Score')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig(save_path)


def save_attention_map(encoder_attn, decoder_attn, cross_attn,
                       inputs, outputs, save_folder):
    encoder_attn = tuple(attn.cpu() for attn in encoder_attn)
    decoder_attn = tuple(attn.cpu() for attn in decoder_attn)
    cross_attn = tuple(attn.cpu() for attn in cross_attn)

    np.save(os.path.join(save_folder, 'encoder_attn'), encoder_attn)
    np.save(os.path.join(save_folder, 'decoder_attn'), decoder_attn)
    np.save(os.path.join(save_folder, 'cross_attn'), cross_attn)
    np.save(os.path.join(save_folder, 'input'), np.array(inputs))
    np.save(os.path.join(save_folder, 'output'), np.array(outputs))


@torch.no_grad()
def visualize_attention(
        args,
        toklen_data,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):

    os.makedirs(args.save_folder, exist_ok=True)
    LOG = logger(name='visualize_attention', log_path=os.path.join(args.save_folder, 'record.log'))
    
    sampler = get_sampler(args, SRC, TRG, toklen_data, scaler, device)

    LOG.info('Save the weights of attention map')
    
    if args.model_type == 'vaetf':
        inputs = SRC.tokenize(args.smiles)
        outputs = ['<sos>'] + inputs + ['<eos>']

        LOG.info(f'Input: ${inputs}')
        LOG.info(f'Output: ${outputs}')

        encoder_attn, decoder_attn, cross_attn = sampler.get_attention_map(args.smiles)

    elif args.model_type == 'scavaetf':
        scaffold = get_scaffold(args.smiles)
        smiles_tokens = SRC.tokenize(args.smiles)

        scaffold = murcko_scaffold(args.smiles)
        scaffold_tokens = SRC.tokenize(scaffold)

        inputs = scaffold_tokens + ['<sep>'] + smiles_tokens
        outputs = ['<sos>'] + inputs + ['<eos>']
        
        LOG.info(f'Input: ${inputs}')
        LOG.info(f'Output: ${outputs}')

        encoder_attn, decoder_attn, cross_attn = sampler.get_attention_map(args.smiles, scaffold)

    save_attention_map(encoder_attn, decoder_attn, cross_attn,
                       inputs, outputs, args.save_folder)

    # visualize the attention map

    LOG.info('Visualize the attention map')

    html_head_view = model_view(
        display_mode='light',
        encoder_attention=encoder_attn,
        decoder_attention=decoder_attn,
        cross_attention=cross_attn,
        encoder_tokens= inputs,
        decoder_tokens = outputs,
        html_action='return'
    )
    
    with open(os.path.join(args.save_folder, 'head_view.html'), 'w') as file:
        file.write(html_head_view.data)