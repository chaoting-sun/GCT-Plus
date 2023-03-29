import os
import numpy as np
import pandas as pd
import torch
# from torchtext import data
from torchtext.data import Example, Dataset
from pathos.multiprocessing import ProcessingPool as Pool

# from Utils.properties import predict_props, property_fn, predict_properties
from Inference.utils import prepare_generator
# from Utils.properties import MurckoScaffoldSimilarity as similarity_fcn
from Utils.properties import tanimoto_similarity as similarity_fcn
from Utils.smiles import is_valid


smiles_list = [
    'Cn1cc(C(=O)N2CCC(c3nc(-c4ccccn4)no3)C2)cn1',
    'Cc1cnc(C(C)NC(=O)CCC(=O)c2ccc(F)c(F)c2)s1',
    'N#Cc1c(Br)cnc(N)c1Br', 
    'CC(=O)Nc1cccc(-c2nc3cc(C)ccc3[nH]c2=O)c1',
    'CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1'
]


prop_list = [
    [1.4948, 75., 0.7250],
    [3., 59.0600, 0.8193],
    [3.06048, 62.70, 0.788971],
    [2.85692, 74.85, 0.762590],
    [2.40440, 89.27, 0.877442]
]


class DataFrameDataset(Dataset):
    def __init__(self, df, SRC, PROP, property_list):
        # get fields for dataset
        fields = [('smiles', SRC)]
        for i, c in enumerate(property_list):
            fields.append((f'src_{c}', PROP[i]))
        for i, c in enumerate(property_list):
            fields.append((f'trg_{c}', PROP[i]))
        
        super(DataFrameDataset, self).__init__(
            [
                Example.fromlist(list(r), fields)
                for i, r in df.iterrows()
            ],
            fields
        )


class PropertyPrediction:
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs

    def get_validity(self, smiles_list):
        with Pool(self.n_jobs) as pool:
            validity_list  = np.array(pool.map(is_valid, smiles_list))
        return validity_list

    def get_similarity(self, smiles_list1, smiles_list2):
        with Pool(self.n_jobs) as pool:
            res = pool.amap(similarity_fcn, smiles_list1, smiles_list2)
        similarity_list = res.get()
        return similarity_list

    def get_conditions(self, smiles_list):
        with Pool(self.n_jobs) as pool:
            conditions_list = np.array(pool.map(predict_props, smiles_list))
        return conditions_list


# with Pool(self.n_jobs) as pool:
#     res = pool.map(smiles_list, )

# def predict_properties(smiles_list, property_list, n_jobs=1):
#     with Pool(n_jobs) as pool:
#         mol_list = list(pool.map(to_mol, smiles_list))

#     calculated_properties = OrderedDict()
#     for prop in property_list:
#         with Pool(n_jobs) as pool:
#             calculated_properties[prop] = list(pool.map(property_fcn[prop], mol_list))
#     return pd.DataFrame(calculated_properties)



class SrcGeneration:
    def __init__(self, args, SRC, PROP, generator, train_smiles, LOG, device,
                 similarity_threshold=0.50, n_samples_per_src=200):
        self.LOG = LOG
        self.SRC = SRC
        self.PROP = PROP
        self.device = device
        self.generator = generator
        self.train_smiles = train_smiles
        self.similarity_threshold = similarity_threshold
        self.n_samples_per_src = n_samples_per_src

        self.n_jobs = args.n_jobs
        self.property_list = args.property_list

        self.n_steps = args.n_steps
        self.n_samples = args.n_samples
        self.n_selections = args.n_selections

        self.predictor = PropertyPrediction(args.n_jobs)


    def prepare_dataset(self, src, props_s, props_t):
        raw_data = pd.DataFrame({ 'src': src })

        src_p = {}
        for i, p in enumerate(self.property_list):
            src_p[f'src_{p}'] = [props_s[0, i]]*len(src)
        src_p = pd.DataFrame(src_p)
        trg_p = {}
        for i, p in enumerate(self.property_list):
            trg_p[f'trg_{p}'] = [props_t[0, i]]*len(src)
        trg_p = pd.DataFrame(trg_p)
        raw_data = pd.concat([raw_data, src_p, trg_p], axis=1)
        return DataFrameDataset(raw_data, self.SRC, self.PROP, self.property_list)


    def preprocess_input_data(self, data):
        src = torch.LongTensor([[self.SRC.vocab.stoi[t]
                                 for t in data.smiles]])
        props_s = np.zeros((1,3))
        props_t = np.zeros((1,3))
        for i, c in enumerate(self.property_list):
            props_s[0, i] = getattr(data, f'src_{c}')
            props_t[0, i] = getattr(data, f'trg_{c}')
        return src, props_s, props_t


    def augment_latent_space(self, mu, std, ep_mean=0, ep_std=1):
        zs = torch.tile(mu, (self.n_samples_per_src, 1, 1))
        for i in range(self.n_samples_per_src):
            eps = torch.empty_like(mu).normal_(ep_mean, ep_std)
            zs[i, :, :] = eps.mul(std).add(mu)
        return zs


    def sample_by_one_step(self, dataset):
        src_list, gen_list = [], []
        self.LOG.info(f'# source smiles: {len(dataset)}')

        for i, data in enumerate(dataset):
            self.LOG.info(f'sample smiles {i} from {"".join(data.smiles)}')

            src, props_s, props_t = self.preprocess_input_data(data)

            _, mu, logvar = self.generator.encode_smiles(src, props_s)
            std = torch.exp(0.5*logvar)
            
            # augment input data
            props_t = np.tile(props_t, (self.n_samples_per_src, 1))
            zs = self.augment_latent_space(mu, std)

            gen = []
            n_per_prediction = 1000
            n_cur_start, n_cur_end = 0, n_per_prediction

            while n_cur_start < self.n_samples_per_src:
                smiles_list, *_ = self.generator.sample_smiles(
                    props_t[n_cur_start:n_cur_end],
                    zs[n_cur_start:n_cur_end]
                )

                n_cur_start += n_per_prediction
                if n_cur_start + n_per_prediction < self.n_samples_per_src:
                    n_cur_end = n_cur_start + n_per_prediction
                else:
                    n_cur_end = self.n_samples_per_src
                gen.extend(smiles_list)
            
            # get unique smiles
            gen = list(set(gen))

            # filter those invalid
            validity = self.predictor.get_validity(gen)
            gen = [g for i, g in enumerate(gen) if validity[i] == 1]

            src_list.append("".join(data.smiles))
            gen_list.append(gen)

        return src_list, gen_list


    def save_predictions(self, src_list, gen_list, i_step, save_folder):
        if i_step == 1:
            prev_prop = predict_properties(src_list, self.property_list, self.n_jobs)
            prev_gen = pd.DataFrame({
                f'smiles_{i_step-1}'    : src_list,
                f'similarity_{i_step-1}': [np.nan],
            })
            prev_gen = pd.concat([prev_gen, prev_prop], axis=1)
            print('src_gen:', prev_gen)
        else:
            prev_gen = pd.read_csv(os.path.join(save_folder, f'gen_{i_step-1}.csv'), index_col=[0])
        prev_gen = prev_gen.rename(columns={f'{p}': f'{p}_{i_step-1}' for p in self.property_list})

        for src_no, src in enumerate(src_list):
            # get similarity and filter smiles by a similarity threshold
            cur_src_list = [src]*len(gen_list[src_no])
            cur_gen_list = gen_list[src_no]

            cur_sim_list = self.predictor.get_similarity(cur_src_list, cur_gen_list)
            
            high_sim_list, high_sim_gen_list = [], []
            for i in range(len(cur_sim_list)):
                if cur_sim_list[i] > self.similarity_threshold:
                    high_sim_list.append(cur_sim_list[i])
                    high_sim_gen_list.append(cur_gen_list[i])
            
            # get properties of the filtered generated smiles

            if src_no == 0:
                total_gen_data = None

            if high_sim_gen_list:
                gen_prop = predict_properties(high_sim_gen_list,
                                              self.property_list,
                                              self.n_jobs)
                gen_data = pd.DataFrame({
                    f'smiles_{i_step-1}'  : [src]*len(high_sim_gen_list),
                    f'smiles_{i_step}'    : high_sim_gen_list,
                    f'similarity_{i_step}': high_sim_list,
                })
                gen_data = pd.concat([gen_data, gen_prop], axis=1)                
                gen_data = gen_data.rename(columns={f'{p}': f'{p}_{i_step}'
                                                    for p in self.property_list})

                gen_data = prev_gen.iloc[[src_no]].merge(
                    gen_data, how='right', on=f'smiles_{i_step-1}'
                )
                total_gen_data = pd.concat([total_gen_data, gen_data], axis=0)

        try:
            total_gen_data = total_gen_data.reset_index(drop=True)
            total_gen_data.to_csv(os.path.join(save_folder, f'gen_{i_step}.csv'))
            print(os.path.join(save_folder, f'gen_{i_step}.csv'))
        except:
            print('No generated data with high similarity!')


    def select_src_smiles(self, i_step, save_folder):
        prev_gen = pd.read_csv(os.path.join(save_folder,
                               f'gen_{i_step-1}.csv'),
                               index_col=[0])
        return prev_gen[f'smiles_{i_step-1}'].tolist()


    def multi_step_sampling(self, src, props_t, n_step, save_folder):
        src = [src]
        props_s = predict_properties(src, self.property_list, self.n_jobs).to_numpy()
        props_t = np.array(props_t)
        props_t_cur = props_s
        props_d = (props_t-props_s)/n_step

        trg_props = np.empty((n_step+1, len(self.property_list)))
        trg_props[0, :] = props_s

        for i in range(1, n_step+1): # from 1 to n_step
            if i > 1:
                src = self.select_src_smiles(i, save_folder)

            props_s_cur = props_t_cur
            props_t_cur = props_s_cur + props_d
            trg_props[i, :] = props_t_cur
    
            self.LOG.info(f'# steps: {n_step}, current step: {i}, Psrc: {props_s_cur}, Ptrg: {props_t_cur}')
            
            dataset = self.prepare_dataset(src, props_s_cur, props_t_cur)
            src_list, gen_list = self.sample_by_one_step(dataset)

            self.save_predictions(src_list, gen_list, i, save_folder)

        trg_props = pd.DataFrame(
            data=trg_props,
            columns=self.property_list,
            index=[f'step_{i}' for i in range(n_step+1)]
        )
        trg_props.to_csv(os.path.join(save_folder, 'trg_props.csv'))


    def generate(self, src, props_t, save_folder):
        for n_step in self.n_steps:
            step_save_folder = os.path.join(save_folder, f"{n_step}_step")
            os.makedirs(step_save_folder, exist_ok=True)
            
            self.multi_step_sampling(src, props_t, n_step, step_save_folder)



smiles_list = [
    'Cn1cc(C(=O)N2CCC(c3nc(-c4ccccn4)no3)C2)cn1',
    'Cc1cnc(C(C)NC(=O)CCC(=O)c2ccc(F)c(F)c2)s1',
    'N#Cc1c(Br)cnc(N)c1Br', 
    'CC(=O)Nc1cccc(-c2nc3cc(C)ccc3[nH]c2=O)c1',
    'CC(NC(=O)OC(C)(C)C)c1nc(CO)nn1Cc1ccccc1'
]

# prop_list = [
#     [1.4948, 89.940, 0.7250],
#     [3.5700, 59.0600, 0.8193],
#     [2.06048, 62.70, 0.788971],
#     [2.85692, 74.85, 0.762590],
#     [2.40440, 89.27, 0.877442]
# ]

prop_list = [
    [1.4948, 75., 0.7250],
    [3., 59.0600, 0.8193],
    [3.06048, 62.70, 0.788971],
    [2.85692, 74.85, 0.762590],
    [2.40440, 89.27, 0.877442]
]


def fast_src_generation(args, toklen_data, train_smiles, 
                        scaler, SRC, TRG, PROP, device, logger):
    for epoch in args.epoch_list:
        # args.src_smiles = smiles_list[i]
        # args.trg_props = prop_list[i]
        
        # args.use_model_path = '/fileserver-gamma/chaoting/ML/molGCT/molgct.pt'
        args.model_path = os.path.join(args.train_path,
                                       args.model_name,
                                       f'model_{epoch}.pt')
        # args.model_path = os.path.join(args.model_folder,
        #                                f'model_{epoch}.pt')
        generator = prepare_generator(args, SRC, TRG, toklen_data,
                                      scaler, device)

        # args.model_name = 'molGCT'

        save_folder = os.path.join(args.inference_path,
                                    'src_generation', 
                                    args.model_name,
                                    str(epoch),
                                    f'{args.src_smiles}'
                                    )
        os.makedirs(save_folder, exist_ok=True)
        LOG = logger(name='src generation',
                     log_path=os.path.join(save_folder, "records.log"))
        LOG.info(args)
        
        scgn = SrcGeneration(args, SRC, PROP, generator, train_smiles, LOG, device)
        scgn.generate(args.src_smiles, args.trg_props, save_folder)


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