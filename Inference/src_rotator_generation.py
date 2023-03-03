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


class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__(
            [
                Example.fromlist(list(r), fields)
                for i, r in df.iterrows()
            ],
            fields
        )


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

        self.n_steps = args.n_steps
        self.n_samples = args.n_samples
        self.n_selections = args.n_selections
        
        self.fields = self._get_fields()

    def _predict_props(self, smiles):
        with Pool(self.n_jobs) as pool:
            props_s = pool.map(predict_props, smiles)
        return np.array(props_s)

    def _get_fields(self):
        fields = [('smiles', self.SRC)]
        for i, c in enumerate(self.conditions):
            fields.append((f'{c}_s', self.COND[i]))
        for i, c in enumerate(self.conditions):
            fields.append((f'{c}_t', self.COND[i]))
        return fields

    def _prepare_dataset(self, src, props_s, props_t):
        raw_data = pd.DataFrame({
            'src'     : src,
            'logp_s'  : [props_s[0, 0]]*len(src),
            'tpsa_s'  : [props_s[0, 1]]*len(src),
            'qed_s'   : [props_s[0, 2]]*len(src),
            'logp_t'  : [props_t[0, 0]]*len(src),
            'tpsa_t'  : [props_t[0, 1]]*len(src),
            'qed_t'   : [props_t[0, 2]]*len(src),
        })
        return DataFrameDataset(df=raw_data, fields=self.fields)

    def _wrap_input(self, data):
        src = torch.LongTensor([[self.SRC.vocab.stoi[t]
                                 for t in data.smiles]])        
        props_s = np.zeros((1,3))
        props_t = np.zeros((1,3))
        for i, c in enumerate(self.conditions):
            props_s[0, i] = getattr(data, f'{c}_s')
            props_t[0, i] = getattr(data, f'{c}_t')
        return src, props_s, props_t

    def _aug_input(self, props, z, zstd=0):
        zs = []
        for i in range(self.n_samples):
            zs.append(z + torch.empty_like(z).normal_(mean=0, std=zstd))
        props = np.tile(props, (self.n_samples, 1))
        return props, zs

    def one_step_sampling(self, dataset):
        src_list, gen_list = [], []
        
        for i, data in enumerate(dataset):
            src, props_s, props_t = self._wrap_input(data)
            for j in range(self.n_samples):
                z, mu, logvar = self.generator.encode_rotator_based_smiles(src, props_s, props_t)
                # props_t, z = self._aug_input(props_t, z)
            
                # std = torch.exp(0.5*logvar)
                # eps = torch.empty_like(std).normal_(mean=0, std=0.8)
                # pseudo_z = eps.mul(std).add(mu)

                smiles, *_ = self.generator.sample_smiles(props_t, z)
                gen_list.append(smiles[0])
                print(j, smiles[0])
    
            src_smi = "".join(data.smiles)
            src_list.extend([src_smi]*self.n_samples)
        return src_list, gen_list

    def _save_props(self, src, gen, props_s, props_t, save_path):
        with Pool(self.n_jobs) as pool:
            props_p = np.array(pool.map(predict_props, gen))
        valid = [0 if np.nan in p else 1 for p in props_p]
        
        with Pool(self.n_jobs) as pool:
            res = pool.amap(similarity_fcn, src, gen)
        similarity = res.get()

        smiles_props = pd.DataFrame({
            "src"       : src, 
            'gen'       : gen,
            'valid'     : valid,
            'similarity': similarity,
            'logp_s'    : [props_s[0, 0]]*len(src),
            'tpsa_s'    : [props_s[0, 1]]*len(src),
            'qed_s'     : [props_s[0, 2]]*len(src),
            'logp_t'    : [props_t[0, 0]]*len(src),
            'tpsa_t'    : [props_t[0, 1]]*len(src),
            'qed_t'     : [props_t[0, 2]]*len(src),
            'logp_p'    : props_p[:, 0],
            'tpsa_p'    : props_p[:, 1],
            'qed_p'     : props_p[:, 2],

        })
        smiles_props.to_csv(save_path)
        return smiles_props
    
    def _select_smiles(self, smiles_props):
        smiles_props = smiles_props.drop_duplicates(subset=['gen'])
        smiles_props = smiles_props[smiles_props.valid == 1]
        smiles_props = smiles_props.sort_values(by=['similarity'],
                                                ascending=False) 

        if len(smiles_props) < self.n_selections:
            n_left = self.n_selections - len(smiles_props)
            s_id = np.random.choice([i for i in range(len(smiles_props))], n_left)
            smiles_props_left = smiles_props.iloc[s_id]
            smiles_props = pd.concat([smiles_props, smiles_props_left],
                                      ignore_index=True, axis=0)
        return smiles_props['gen'].iloc[:self.n_selections].tolist()

    def multi_step_sampling(self, src, props_t, n_step, step_save_folder):
        src = [src]
        props_s = self._predict_props(src)
        props_t = np.array(props_t)
        props_t_cur = props_s
        props_d = (props_t-props_s)/n_step

        for i in range(n_step):
            self.LOG.info(f'# steps: {n_step}, curr step: {i+1}')
            props_s_cur = props_t_cur
            props_t_cur = props_s_cur + props_d
            
            self.LOG.info(f'src properties = logP: {props_s_cur[0,0]:.2f}, '
                                            f'tPSA: {props_s_cur[0,1]:.2f}, '
                                            f'QED:  {props_s_cur[0,2]:.2f}')
            print(f'trg properties = logP: {props_t_cur[0,0]:.2f}, '
                                   f'tPSA: {props_t_cur[0,1]:.2f}, '
                                   f'QED:  {props_t_cur[0,2]:.2f}')
            print('src smiles:', src)
            
            dataset = self._prepare_dataset(src, props_s_cur, props_t_cur)
            src, gen = self.one_step_sampling(dataset)

            print('gen smiles:', gen)

            smiles_props = self._save_props(src, gen, props_s_cur, props_t_cur,
                os.path.join(step_save_folder, f"{i+1}.csv"))
            print('save path:', os.path.join(step_save_folder, f"{i+1}.csv"))

            src = self._select_smiles(smiles_props)


    def generate(self, src, props_t, save_folder):
        for n_step in self.n_steps:
            step_save_folder = os.path.join(save_folder, f"{n_step}_step")
            os.makedirs(step_save_folder, exist_ok=True)
            
            self.multi_step_sampling(src, props_t, n_step, step_save_folder)
            
            
def fast_src_rotator_generation(args, toklen_data, train_smiles,
                                scaler, SRC, TRG, COND, device, logger):

    for epoch in args.epoch_list:
        args.use_model_path = os.path.join(args.train_path,
                                           args.model_name,
                                           f'model_{epoch}.pt')
        generator = prepare_generator(args, SRC, TRG,
                                      toklen_data, scaler, device)

        # args.model_name = 'molGCT'

        save_folder = os.path.join(args.inference_path, 'src_generation', 
                                   args.model_name, str(epoch))
        os.makedirs(save_folder, exist_ok=True)
        
        LOG = logger(name='src generation',
                     log_path=os.path.join(save_folder, "records.log"))

        scgn = SrcGeneration(args, SRC, COND, generator, train_smiles, LOG, device)
        scgn.generate(args.src_smiles, args.trg_props, save_folder)

