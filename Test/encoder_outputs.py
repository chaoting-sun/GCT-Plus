import os
import torch
import numpy as np
import pandas as pd
from torchtext import data

from Utils.seed import set_seed
from Utils.field import get_tf_fields
from Utils.dataset import get_loader
from Inference.utils import prepare_generator
from plot import density_plot, density_plot_dict
import matplotlib.pyplot as plt
import seaborn as sns


class EncoderOutput(object):
    def __init__(self, args, scaler, device, toklen_data, data_type):
        self.args = args
        self.epoch_list = args.epoch_list
        self.data_path = args.data_path
        self.max_strlen = args.max_strlen
        self.latent_dim = args.latent_dim

        self.fields, self.SRC, self.TRG = get_tf_fields(args.conditions, args.molgct_path)
        self.pad_id = self.SRC.vocab.stoi['<pad>']
        
        self.scaler = scaler
        self.device = device
        self.toklen_data = toklen_data
        self.data_type = data_type
        
        self.n_data = 100 if args.debug else 10000
        self.n_sample = 10 if args.debug else 1000

        self.require_neq_data = True if args.similarity != 1 else False
        
        self.data_folder = os.path.join(self.data_path, 'aug', f'data_sim{args.similarity:.2f}_tol{args.tolerance:.2f}')
        self.save_folder = os.path.join(args.train_path, args.model_name, 'encoder_outputs')
        # self.save_folder = '/fileserver-gamma/chaoting/ML/molGCT/encoder_outputs'
        os.makedirs(self.save_folder, exist_ok=True)
        
        
    def create_input_data(self):
        df = pd.read_csv(os.path.join(self.data_folder, f'{self.data_type}.csv'))
        df_eq, df_neq = df.loc[df.src == df.trg], df.loc[df.src != df.trg]
        
        df_eq = df_eq.sample(self.n_data, random_state=1, ignore_index=True)
        if self.require_neq_data:
            df_neq = df_neq.sample(self.n_data, random_state=1, ignore_index=True)
            df = pd.concat([df_eq, df_neq], axis=0)
        else:
            df = df_eq
        df.to_csv(os.path.join(self.data_folder, f'{self.data_type}_sample.csv'), index=False)


    def prepare_iterator(self, batch_size=256):
        dataset = data.TabularDataset(
            path=os.path.join(self.data_folder, f'{self.data_type}_sample.csv'),
            format='csv',
            fields=self.fields,
            skip_header=True
        )
        data_iter = data.BucketIterator(
            dataset=dataset,
            batch_size=batch_size,
            sort_key=lambda x: (len(x.src), len(x.trg))
        )
        return data_iter
    
        
    def get_encoder_outputs(self, generator, dataloader):
        
        out_dim = (2*self.n_data if self.require_neq_data else self.n_data,
                   self.max_strlen, self.latent_dim)
        zs, mus, stds = np.empty(out_dim), np.empty(out_dim), np.empty(out_dim)
        
        n_acc = 0
        
        for i, batch in enumerate(dataloader):
            print(i)
            z, mu, logvar = generator.encode_smiles(batch.src,
                                                    batch.econds,
                                                    transform=False)
            std = torch.exp(0.5*logvar)

            n_cur = z.size(0)
            
            zs[n_acc:n_acc+n_cur, :, :] = z.cpu().numpy()
            mus[n_acc:n_acc+n_cur, :, :] = mu.cpu().numpy()
            stds[n_acc:n_acc+n_cur, :, :] = std.cpu().numpy()

            n_acc += n_cur
            
        return zs, mus, stds
        
    
    def save_encoder_outputs(self, outs, data_name):
        """mu, logvar, z"""
        
        x = np.tile([i for i in range(self.latent_dim)],
                    len(outs)*self.max_strlen)
        y = np.reshape(outs, (-1,))

        ids = np.random.choice(np.arange(len(x)), self.n_sample, replace=False)
        
        plot_dict = density_plot_dict(
            xlabel='dimension of latent space',
            ylabel='position',
            nbins=128,
            figsize=None,
            xlim=None,
            ylim=(-3, 3)
        )
        density_plot(x[ids], y[ids], os.path.join(self.save_folder, f'{data_name}.png'), plot_dict)
        print(f'{data_name} - '
              f'min: {outs.min():.2f}, max: {outs.max():.2f}, '
              f'mean: {outs.mean():.2f}, std: {y.std():.2f}')
    
    
    def get_statistics(self, outs):
        return {
            'min' : outs.min(),
            'max' : outs.max(),
            'mean': outs.mean(),
            'std' : outs.std(),
        }


    def plot_correlation_figure(self, data, fig_name):
        corr = data.corr()
        
        plt.figure()
        ax = sns.heatmap(
            corr,
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True
        )
        ax.set_xlabel('dimension of latent space', fontsize=12)
        ax.set_ylabel('dimension of latent space', fontsize=12)

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        );
        
        plt.savefig(os.path.join(self.save_folder, fig_name), bbox_inches="tight")
            

    def save_string_len_correlation(self, outs, data_name):
        """
        x, y: (bs*latent_dim, max_strlen)
        """
        
        x = np.tile([i for i in range(self.latent_dim)], len(outs))
        outs = np.transpose(outs, (0, 2, 1)) 
        y = np.reshape(outs, (-1, self.max_strlen))
        y = pd.DataFrame(y, columns=[i for i in range(self.max_strlen)])

        self.plot_correlation_figure(y, f'{data_name}_strlen_corr.png')


    def save_latent_dim_correlation(self, outs, data_name):
        """
        x, y: (bs*max_strlen, latent_dim)
        """
        x = np.tile([i for i in range(self.max_strlen)], len(outs))
        y = np.reshape(outs, (-1,self.latent_dim))

        ids = np.random.choice(np.arange(len(x)), self.n_sample, replace=False)
        x, y = x[ids], y[ids]
        y = pd.DataFrame(y, columns=[i for i in range(self.latent_dim)])

        self.plot_correlation_figure(y, f'{data_name}_latdim_corr.png')
            
    
    def runner(self, args, logger):
        set_seed(0)
        self.create_input_data()        

        LOG = logger('EncoderOutput', os.path.join(self.save_folder, "records.log"))
        
        data_iter = self.prepare_iterator()

        z_stat_dict = { 'min': [], 'max': [], 'mean': [], 'std': [] }
        mu_stat_dict = { 'min': [], 'max': [], 'mean': [], 'std': [] }
        std_stat_dict = { 'min': [], 'max': [], 'mean': [], 'std': [] }

        with torch.no_grad():
            for epoch in self.epoch_list:
                LOG.info(f"model epoch: {epoch}")
                args.use_model_path = os.path.join(args.train_path,
                    args.model_name, f'model_{epoch}.pt')
                # args.use_model_path = '/fileserver-gamma/chaoting/ML/molGCT/molgct.pt'
                
                generator = prepare_generator(args, self.SRC, self.TRG,
                    self.toklen_data, self.scaler, self.device)
                
                dataloader = get_loader(data_iter, args.conditions,
                    self.pad_id, args.max_strlen, args.pad_to_same_len)

                zs, mus, stds = self.get_encoder_outputs(generator, dataloader)
   
                self.save_latent_dim_correlation(zs, 'zs')
                self.save_latent_dim_correlation(mus, 'mus')
                self.save_latent_dim_correlation(stds, 'stds')
   
                self.save_string_len_correlation(zs, 'zs')
                self.save_string_len_correlation(mus, 'mus')
                self.save_string_len_correlation(stds, 'stds')
            
                self.save_encoder_outputs(zs, 'zs')
                self.save_encoder_outputs(mus, 'mus')
                self.save_encoder_outputs(stds, 'stds')
                
                for stat, val in self.get_statistics(zs).items():
                    z_stat_dict[stat].append(val)

                for stat, val in self.get_statistics(mus).items():
                    mu_stat_dict[stat].append(val)

                for stat, val in self.get_statistics(stds).items():
                    std_stat_dict[stat].append(val)

        df = pd.DataFrame(data=z_stat_dict, index=self.epoch_list)
        df.to_csv(os.path.join(self.save_folder, 'z_stat.csv'))
                
        df = pd.DataFrame(data=mu_stat_dict, index=self.epoch_list)
        df.to_csv(os.path.join(self.save_folder, 'mu_stat.csv'))

        df = pd.DataFrame(data=std_stat_dict, index=self.epoch_list)
        df.to_csv(os.path.join(self.save_folder, 'std_stat.csv'))


def encoder_outputs(args, toklen_data, scaler, device, logger, data_type='train'):
    eoObj = EncoderOutput(args, scaler, device, toklen_data, data_type)
    eoObj.runner(args, logger)


# def create_input_data(data_folder, in_path, out_path,
#                       n=10000, require_neq=False):
#     df = pd.read_csv(os.path.join(data_folder, in_path))
#     df_eq, df_neq = df.loc[df.src == df.trg], df.loc[df.src != df.trg]
    
#     df_eq = df_eq.sample(n, random_state=1, ignore_index=True)
#     if require_neq:
#         df_neq = df_neq.sample(n, random_state=1, ignore_index=True)
#         df = pd.concat([df_eq, df_neq], axis=0)
#     else:
#         df = df_eq
#     df.to_csv(os.path.join(data_folder, out_path), index=False)


# def prepare_iterator(data_path, fields, batch_size=256):
#     dataset = data.TabularDataset(path=data_path,
#                                   format='csv',
#                                   fields=fields,
#                                   skip_header=True
#                                   )
#     data_iter = data.BucketIterator(dataset=dataset,
#                                     batch_size=batch_size,
#                                     sort_key=lambda x: (len(x.src), len(x.trg)))
#     return data_iter, len(dataset)


# def get_latent_space(dataloader, generator, max_strlen,
#                      latent_dim, n_data):
#     zs = np.empty((n_data, max_strlen, latent_dim))
#     mus = np.empty((n_data, max_strlen, latent_dim))
#     stds = np.empty((n_data, max_strlen, latent_dim))
    
#     n_acc = 0
    
#     for i, batch in enumerate(dataloader):
#         print(i)
#         z, mu, logvar = generator.encode_smiles(batch.src,
#                                                 batch.econds,
#                                                 transform=False)
#         std = torch.exp(0.5*logvar)

#         n_cur = z.size(0)
        
#         zs[n_acc:n_acc+n_cur, :, :] = z.cpu().numpy()
#         mus[n_acc:n_acc+n_cur, :, :] = mu.cpu().numpy()
#         stds[n_acc:n_acc+n_cur, :, :] = std.cpu().numpy()

#         n_acc += n_cur
#     return zs, mus, stds
    

# def plot_density_figure(x, y, fig_path, n=10000):
#     ids = np.random.choice(np.arange(len(x)), n, replace=False)
    
#     plot_dict = density_plot_dict(
#         xlabel='dimension of latent space',
#         ylabel='position',
#         nbins=128,
#         figsize=None,
#         xlim=None,
#         ylim=None
#     )
    
#     density_plot(x[ids], y[ids], fig_path, plot_dict)


# def encoder_outputs(args, toklen_data, scaler, device, logger, data_type='train'):
#     set_seed(0)
    
#     save_folder = os.path.join(args.train_path,
#                                args.model_name,
#                                'reconstruction')
#     os.makedirs(save_folder, exist_ok=True)

#     LOG = logger(name='augment data by conditions',
#                 log_path=os.path.join(save_folder, "records.log"))
#     fields, SRC, TRG = get_tf_fields(args.conditions, args.molgct_path)
#     args.pad_id = SRC.vocab.stoi['<pad>']

#     data_folder = os.path.join(args.data_path, 'aug',
#                              f'data_sim{args.similarity:.2f}_tol{args.tolerance:.2f}')
#     if args.similarity == 1:
#         create_input_data(data_folder, f'{data_type}.csv', f'{data_type}_sample.csv')
#     else:
#         create_input_data(data_folder, f'{data_type}.csv', f'{data_type}_sample.csv', True)
#     data_iter, n_data = prepare_iterator(os.path.join(data_folder, 'valid_sample.csv'), fields)


#     with torch.no_grad():
#         for epoch in args.epoch_list:
#             LOG.info(f"model epoch: {epoch}")

#             args.use_model_path = os.path.join(args.train_path,
#                                                args.model_name,
#                                                f'model_{epoch}.pt')
#             # args.use_model_path = '/fileserver-gamma/chaoting/ML/molGCT/molgct.pt'
#             generator = prepare_generator(args,
#                                           SRC,
#                                           TRG,
#                                           toklen_data,
#                                           scaler,
#                                           device
#                                           )
            
#             dataloader = get_loader(data_iter,
#                                     args.conditions,
#                                     args.pad_id,
#                                     args.max_strlen,
#                                     args.pad_to_same_len
#                                     )

#             zs, mus, stds = get_latent_space(dataloader,
#                                              generator,
#                                              args.max_strlen,
#                                              args.latent_dim,
#                                              n_data
#                                              )
                        

#             plot_density_figure(x, y, './zs.png')
#             print('z:', y.min(), y.max(), y.mean(), y.std())
            
#             x = np.tile([i for i in range(args.latent_dim)],
#                         len(mus)*args.max_strlen)
#             y = np.reshape(mus, (-1))
#             plot_density_figure(x, y, './mus.png')
#             print('mu:', y.min(), y.max(), y.mean(), y.std())

#             x = np.tile([i for i in range(args.latent_dim)],
#                         len(stds)*args.max_strlen)
#             y = np.reshape(stds, (-1))
#             plot_density_figure(x, y, './stds.png')
#             print('std:', y.min(), y.max(), y.mean(), y.std())
