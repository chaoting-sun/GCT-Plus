import os
import numpy as np
import pandas as pd
import torch
# from torchtext import data
from torchtext.data import Example, Dataset
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.properties import predict_props
from Inference.utils import prepare_generator
from Utils.properties import tanimoto_similarity as similarity_fcn
from itertools import combinations


class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__(
            [
                Example.fromlist(list(r), fields)
                for i, r in df.iterrows()
            ],
            fields
        )
        

def get_rand_combs(total_choices, n_choices, n_samples=200):
    np.random.seed(0)
    total_choice_list = np.arange(total_choices)

    if n_choices in (1, 2, 3):
        comb_list = list(combinations(total_choice_list, n_choices))
        np.random.shuffle(comb_list)
    else:
        comb_set = set()
        while len(comb_set) < n_samples:
            choices = np.random.choice(total_choice_list, n_choices, replace=False)
            comb_set.add(tuple(choices))
        comb_list = list(comb_set)
    
    if len(comb_list) < n_samples:
        return comb_list
    else:
        return comb_list[:n_samples]


class SrcGeneration:
    def __init__(self, args, SRC, COND, generator,
                 train_smiles, LOG, device):
        self.LOG = LOG
        self.SRC = SRC
        self.COND = COND
        self.device = device
        self.generator = generator
        self.train_smiles = train_smiles
        self.n_jobs = args.n_jobs
        self.conditions = args.conditions
        self.model_type = args.model_type

        if hasattr(args, 'n_vars'):
            self.n_vars = args.n_vars
        if hasattr(args, 'n_fixed'):
            self.n_fixed = args.n_fixed
        
        self.n_steps = args.n_steps
        self.n_samples = args.n_samples
        self.latent_dim = args.latent_dim
        # self.n_selections = args.n_selections
        
        self.fields = self._get_fields()


    def _get_fields(self):
        fields = [('smiles', self.SRC)]
        for i, c in enumerate(self.conditions):
            fields.append((f'{c}_s', self.COND[i]))
        for i, c in enumerate(self.conditions):
            fields.append((f'{c}_t', self.COND[i]))
        return fields
    

    def prepare_dataset(self, src, props_s, props_t):
        raw_data = pd.DataFrame(
            data=np.concatenate((src, props_s, props_t), axis=1),
            columns=['src', 'logp_s', 'tpsa_s', 'qed_s',
                     'logp_t', 'tpsa_t', 'qed_t']
        )
        return DataFrameDataset(df=raw_data, fields=self.fields)

    def preprocess_input_data(self, data):
        src = torch.LongTensor([[self.SRC.vocab.stoi[t]
                                 for t in data.smiles]])
        props_s = np.zeros((1,3))
        props_t = np.zeros((1,3))
        for i, c in enumerate(self.conditions):
            props_s[0, i] = getattr(data, f'{c}_s')
            props_t[0, i] = getattr(data, f'{c}_t')
        return src, props_s, props_t


    def aug_z_with_var_ids(self, mu_outs, std_outs, var_ids, mean=0, std=1):
        crafted_z = torch.tile(mu_outs, (self.n_samples, 1, 1))
        for i in range(self.n_samples):
            eps = torch.empty_like(mu_outs).normal_(mean, std)
            z = eps.mul(std_outs).add(mu_outs)
            crafted_z[i, :, var_ids] = z[:, :, var_ids]
        return crafted_z
        
    
    def aug_z_with_fixed_ids(self, mu_outs, std_outs, fixed_ids):
        ld_ids = np.arange(self.latent_dim)        
        var_ids = np.array([i for i in ld_ids if i not in fixed_ids])
        return self.aug_z_with_var_ids(mu_outs, std_outs, var_ids)


    def optimize_smiles_by_one_step(self, dataset, var_ids):
        gen_list = []

        for i, data in enumerate(dataset):
            src, props_s, props_t = self.preprocess_input_data(data)
            _, mu, logvar = self.generator.encode_smiles(src, props_s)
            std = torch.exp(0.5*logvar)
            
            if self.n_vars:
                crafted_z = self.aug_z_with_var_ids(mu, std, self.)
            
            crafted_z = self.aug_z_by_ids_of_one_src(mu, std, var_ids)
            props_t_list = np.tile(props_t, (self.n_samples, 1))
            smiles, *_ = self.generator.sample_smiles(props_t_list, crafted_z)
            print(smiles)
            gen_list.extend(smiles)
        return gen_list


    def get_properties(self, smiles_list):
        with Pool(self.n_jobs) as pool:
            smiles_list = np.array(pool.map(predict_props, smiles_list))
        return smiles_list
    

    def get_src_properties(self, src):
        return self.get_properties(src)
    

    def get_gen_properties(self, src, gen_list):
        props_p = self.get_properties(gen_list)
        valid_list = [0 if np.nan in p else 1 for p in props_p]
        with Pool(self.n_jobs) as pool:
            res = pool.amap(similarity_fcn,
                            [src]*len(gen_list),
                            gen_list)
        sim_list = res.get()
        return props_p, valid_list, sim_list


    def save_gen_properties(self, src, gen_list,
                            props_s, props_t, props_p,
                            valid_list, sim_list, save_path):
        smiles_props = pd.DataFrame({
            'src'       : [src]*len(gen_list),
            'gen'       : gen_list,
            'valid'     : valid_list,
            'similarity': sim_list,
            'logp_s'    : [props_s[0, 0]]*len(gen_list),
            'tpsa_s'    : [props_s[0, 1]]*len(gen_list),
            'qed_s'     : [props_s[0, 2]]*len(gen_list),
            'logp_t'    : [props_t[0, 0]]*len(gen_list),
            'tpsa_t'    : [props_t[0, 1]]*len(gen_list),
            'qed_t'     : [props_t[0, 2]]*len(gen_list),
            'logp_p'    : props_p[:, 0],
            'tpsa_p'    : props_p[:, 1],
            'qed_p'     : props_p[:, 2],
        })
        smiles_props.to_csv(save_path)
        return smiles_props


    # def _select_smiles(self, smiles_props):
    #     smiles_props = smiles_props.drop_duplicates(subset=['gen'])
    #     smiles_props = smiles_props[smiles_props.valid == 1]
    #     smiles_props = smiles_props.sort_values(by=['similarity'],
    #                                             ascending=False) 

    #     if len(smiles_props) < self.n_selections:
    #         n_left = self.n_selections - len(smiles_props)
    #         s_id = np.random.choice([i for i in range(len(smiles_props))], n_left)
    #         smiles_props_left = smiles_props.iloc[s_id]
    #         smiles_props = pd.concat([smiles_props, smiles_props_left],
    #                                   ignore_index=True, axis=0)
    #     return smiles_props['gen'].iloc[:self.n_selections].tolist()

    # def multi_step_sampling(self, src, props_t, n_step, step_save_folder):
    #     src = [src]
    #     props_s = self._predict_props(src)
    #     props_t = np.array(props_t)
    #     props_t_cur = props_s
    #     props_d = (props_t-props_s)/n_step

    #     for i in range(n_step):
    #         self.LOG.info(f'# steps: {n_step}, curr step: {i+1}')
    #         props_s_cur = props_t_cur
    #         props_t_cur = props_s_cur + props_d
            
    #         self.LOG.info(f'src properties = logP: {props_s_cur[0,0]:.2f}, '
    #                                         f'tPSA: {props_s_cur[0,1]:.2f}, '
    #                                         f'QED:  {props_s_cur[0,2]:.2f}')
    #         print(f'trg properties = logP: {props_t_cur[0,0]:.2f}, '
    #                                f'tPSA: {props_t_cur[0,1]:.2f}, '
    #                                f'QED:  {props_t_cur[0,2]:.2f}')
    #         print('src smiles:', src)
            
    #         dataset = self._prepare_dataset(src, props_s_cur, props_t_cur)
    #         src, gen = self.one_step_sampling(dataset)

    #         print('gen smiles:', gen)

    #         smiles_props = self._save_props(src, gen, props_s_cur, props_t_cur,
    #             os.path.join(step_save_folder, f"{i+1}.csv"))
    #         print('save path:', os.path.join(step_save_folder, f"{i+1}.csv"))

    #         src = self._select_smiles(smiles_props)


    # def generate(self, src, props_t, save_folder):
    #     for n_step in self.n_steps:
    #         step_save_folder = os.path.join(save_folder, f"{n_step}_step")
    #         os.makedirs(step_save_folder, exist_ok=True)
            
    #         self.multi_step_sampling(src, props_t, n_step, step_save_folder)


    def sample_smiles(self, src, props_t, comb_list, save_folder):
        var_ids = get_rand_combs(self.latent_dim, self.n_vars)
                       
        src_list = np.array([src]).reshape((-1, 1))
        props_s = self.get_src_properties([src])

        if np.array(props_s).ndim == 1:
            props_s = np.array(props_s).reshape((-1, len(self.conditions)))
        if np.array(props_t).ndim == 1:
            props_t = np.array(props_t).reshape((-1, len(self.conditions)))

        self.LOG.info('prepare dataset...')
        dataset = self.prepare_dataset(src_list, props_s, props_t)

        for var_ids in comb_list:
            file_name = '_'.join(list(map(str, var_ids)))
            save_path = os.path.join(save_folder, f'{file_name}.csv')

            self.LOG.info('optimize smiles by one step...')
            gen_list = self.optimize_smiles_by_one_step(dataset, var_ids)
            props_p, valid_list, sim_list = self.get_gen_properties(src, gen_list)
                
            self.LOG.info('save properties of generated smiles...')
            smiles_props = self.save_gen_properties(src,
                                                    gen_list,
                                                    props_s,
                                                    props_t,
                                                    props_p,
                                                    valid_list,
                                                    sim_list,
                                                    save_path
                                                    )
            print(var_ids)
            print(smiles_props)


def left_anti_join(dfA, dfB, left_on, right_on):
    df = pd.merge(dfA, dfB,  how='outer', left_on=left_on,
                    right_on=right_on, indicator = True)
    df = df.loc[df['_merge'] == 'left_only'].drop('_merge', axis=1)
    return df


class SrcGenStatistics:
    def __init__(self, args, train_smiles):
        self.latent_dim = args.latent_dim
        # self.train_smiles = train_smiles

    def get_new_gen_rows(self, df):
        return df.loc[df.src != df.gen]

    def get_unique_gen_rows(self, df):
        return df.drop_duplicates(subset='gen')

    def get_similar_gen_rows(self, df, threshold):
        return df.loc[df.similarity >= threshold]

    def get_unrepeated_gen_by_dim(self, dim_gen_list, comb_list):
        """
        dim_gen_list: list of DataFrames including all unique gens
        """
        print('get unrepeated gen by dim...')
        unrepeated_dim_gen_list = []
        
        assert len(dim_gen_list) == len(comb_list)
        for i in range(len(comb_list)):
            focused_gen = dim_gen_list[i]

            other_gen = dim_gen_list[:i] + dim_gen_list[i+1:]
            other_gen = pd.concat(other_gen, axis=0)
            other_gen = self.get_unique_gen_rows(other_gen)
            
            comb_cols = ['src', 'gen',
                         'valid', 'similarity',
                         'logp_s', 'tpsa_s', 'qed_s',
                         'logp_t', 'tpsa_t', 'qed_t',
                         'logp_p', 'tpsa_p', 'qed_p',
                         ]
            dim_exclusive_gen = left_anti_join(focused_gen,
                                               other_gen,
                                               left_on=comb_cols,
                                               right_on=comb_cols
                                               )
            print(comb_list[i], len(dim_exclusive_gen))
            # print(dim_exclusive_gen)
            if len(dim_exclusive_gen):
                print(dim_exclusive_gen)
            unrepeated_dim_gen_list.append(dim_exclusive_gen)
        return unrepeated_dim_gen_list


    def runner(self, comb_list, save_folder):
        dim_gen_list = []
        
        for var_ids in comb_list:
            file_name = '_'.join(list(map(str, var_ids)))
            save_path = os.path.join(save_folder, f'{file_name}.csv')

            df = pd.read_csv(save_path, index_col=[0])
            
            df = df.loc[df.valid == 1]
            
            df = self.get_new_gen_rows(df)
            df = self.get_unique_gen_rows(df)
            df = self.get_similar_gen_rows(df, threshold=0.4)

            print(var_ids, len(df))

            dim_gen_list.append(df)

        self.get_unrepeated_gen_by_dim(dim_gen_list, comb_list)


# smiles_list = [
#     'Cn1ncc(Br)c1NC(=O)Nc1ccccc1',
#     'CCN1C(=O)C(O)(CC(=O)c2ccc(C)cc2)c2ccccc21',
#     'O=C(NC1CCc2cc(F)ccc21)c1[nH]nc2c1CCCC2'
# ]


# prop_list = [
#     [2.82660, 58.95, 0.894693],
#     [2.82212, 57.61, 0.883604],
#     [2.84490, 57.78, 0.895593]
# ]


# smiles_list = [
#     'Cn1cc(C(=O)N2CCC(c3nc(-c4ccccn4)no3)C2)cn1',
#     'Cc1cnc(C(C)NC(=O)CCC(=O)c2ccc(F)c(F)c2)s1',
#     'N#Cc1c(Br)cnc(N)c1Br', 
#     'CC(=O)Nc1cccc(-c2nc3cc(C)ccc3[nH]c2=O)c1',
#     'CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1'
# ]

# prop_list = [
#     [1.4948, 89.940, 0.7250],
#     [3.5700, 59.0600, 0.8193],
#     [2.06048, 62.70, 0.788971],
#     [2.85692, 74.85, 0.762590],
#     [2.40440, 89.27, 0.877442]
# ]

# prop_list = [
#     [1.4948, 75., 0.7250],
#     [3., 59.0600, 0.8193],
#     [3.06048, 62.70, 0.788971],
#     [2.85692, 74.85, 0.762590],
#     [2.40440, 89.27, 0.877442]
# ]


            

def fast_src_generation(args, toklen_data, train_smiles, scaler,
                        SRC, TRG, COND, device, logger):
    for epoch in args.epoch_list:
        # args.src_smiles = smiles_list[i]
        # args.trg_props = prop_list[i]
        
        # args.model_name = 'molGCT'
        model_scgn_folder = os.path.join(args.inference_path,
                                         'src_generation',
                                         args.model_name,
                                         str(epoch)
                                         )
        save_n_vars_folder = os.path.join(model_scgn_folder,
                                          args.src,
                                          f'var_{args.n_vars}')
        os.makedirs(save_n_vars_folder, exist_ok=True)
        
        # comb_list = get_random_combinations(args.latent_dim, args.n_vars)
        # sgst = SrcGenStatistics(args, train_smiles)
        # sgst.runner(comb_list, save_n_vars_folder)

        # exit()

        print("prepare generator...")
        # args.use_model_path = '/fileserver-gamma/chaoting/ML/molGCT/molgct.pt'
        
        args.use_model_path = os.path.join(args.train_path,
                                            args.model_name,
                                            f'model_{epoch}.pt')
        generator = prepare_generator(args, SRC, TRG, toklen_data,
                                      scaler, device)


        LOG = logger(name='src generation',
                        log_path=os.path.join(save_n_vars_folder, "records.log"))

        print("prepare srcGeneration object...")
        scgn = SrcGeneration(args, SRC, COND, generator, train_smiles, LOG, device)
        
        comb_list = get_random_combinations(args.latent_dim, args.n_vars)
        scgn.sample_smiles(args.src, args.trg_props, comb_list, save_n_vars_folder)


# from collections import OrderedDict


# def plot_similarity_density(args):
#     """
#     src and trg are the same
#     """
    
#     data_dict = OrderedDict()
    
#     for epoch in args.epoch_list:
#         for i in range(3):
#             args.src_smiles = smiles_list[i]
#             args.trg_props = prop_list[i]
#             save_folder = os.path.join(args.inference_path,
#                                        'src_generation', 
#                                        args.model_name,
#                                        str(epoch),
#                                        smiles_list[i],
#                                        '1_step', '1.csv'
#                                        )
#             df = pd.read_csv(save_folder)
#             df = df.drop_duplicates(subset = "gen")
#             data_dict['CVAE-TF1'] = df
            
#     print(data_dict)
#     exit()