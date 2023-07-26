import os
import torch
from Model.build_model import get_generator
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem


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


def save_attention_map(encoder_attn, decoder_attn_1, decoder_attn_2,
                       inputs, outputs, task_path):
    encoder_attn = tuple(attn.cpu() for attn in encoder_attn)
    decoder_attn_1 = tuple(attn.cpu() for attn in decoder_attn_1)
    decoder_attn_2 = tuple(attn.cpu() for attn in decoder_attn_2)

    np.save(os.path.join(task_path, 'encoder_attn'), encoder_attn)
    np.save(os.path.join(task_path, 'decoder_attn'), decoder_attn_1)
    np.save(os.path.join(task_path, 'cross_attn'), decoder_attn_2)
    np.save(os.path.join(task_path, 'input'), np.array(inputs))
    np.save(os.path.join(task_path, 'output'), np.array(outputs))


@torch.no_grad()
def visualize_attention(
        args,
        toklen_data,
        df_train,
        df_test,
        df_test_scaffolds,
        scaler,
        SRC,
        TRG,
        device,
        logger
    ):

    # task_path = os.path.join(args.infer_path, args.benchmark, 'psca_sampling')
    task_path = '/home/chaoting/ML/cvae-transformer/Experiment/'
    os.makedirs(task_path, exist_ok=True)

    smiles = 'CCc1cc(C(=O)NCc2cccs2)cs1'
    # smiles = 'O=C(Cc1ccccc1)NCc1ccccc1'
    smiles = 'CC(Cc1ccc(c(c1)OC)O)N'


    args.model_path = os.path.join(args.train_path, args.benchmark,
                                   args.model_name, f'model_{args.epoch}.pt')
    sampler = get_generator(args, SRC, TRG, toklen_data, scaler, device)


    if args.model_type == 'vaetf':
        # smiles = 'N1(C(=O)CCSc2oc(CC)nn2)CCCC1'

        smiles = 'NC(=O)c1ccccc1OCC1CC1(Cl)Cl'
        smiles = 'Cn1cc(SCc2cc(Br)cs2)cn1'
        smiles = 'COC(=O)C(NC(=O)OC(C)(C)C)c1cccc(Cl)c1'
        smiles = 'C(Cc1c(OC)c(OC)ccc1Br)C#N'

        inputs = SRC.tokenize(smiles)
        outputs = ['<sos>'] + inputs + ['<eos>']

        encoder_attn, decoder_attn_1, decoder_attn_2 = sampler.get_attention_map(smiles)

    elif args.model_type == 'scavaetf':
        # smiles = 'OSCCSc1nc2cc(Cl)ccc2o1'

        smiles = 'NC(=O)c1ccccc1OCC1CC1(Cl)Cl'
        smiles = 'Cn1cc(SCc2cc(Br)cs2)cn1'
        smiles = 'COC(=O)C(NC(=O)OC(C)(C)C)c1cccc(Cl)c1'
        smiles = 'C(Cc1c(OC)c(OC)ccc1Br)C#N'

        scaffold = get_scaffold(smiles)

        smiles_tokens = SRC.tokenize(smiles)
        scaffold = get_scaffold(smiles)
        scaffold_tokens = SRC.tokenize(scaffold)
        inputs = scaffold_tokens + ['<sep>'] + smiles_tokens
        outputs = ['<sos>'] + inputs + ['<eos>']

        encoder_attn, decoder_attn_1, decoder_attn_2 = sampler.get_attention_map(smiles, scaffold)


    # save_folder = os.path.join(task_path, f'{args.model_name}-{args.epoch}',
    #                            args.sample_from)
    # os.makedirs(save_folder, exist_ok=True)

    print('self-attention of the inputs (each encoderlayer):', encoder_attn[0].size())
    print('self-attention of the outputs (each decoderlayer):', decoder_attn_1[0].size())
    print('attention between the inputs and outputs (each decoderlayer):', decoder_attn_2[0].size())

    if args.model_type == 'vaetf':
        save_path = os.path.join(task_path, 'vaetf')
    elif args.model_type == 'scavaetf':
        save_path = os.path.join(task_path, 'scavaetf')

    os.makedirs(save_path, exist_ok=True)

    save_attention_map(encoder_attn, decoder_attn_1,
                       decoder_attn_2, inputs, outputs,
                       save_path)

