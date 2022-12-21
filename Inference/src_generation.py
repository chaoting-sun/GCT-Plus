import os
import numpy as np
import pandas as pd
import torch
# from torchtext import data
from torchtext.data import Example, Dataset
from rdkit.Chem import MolFromSmiles
from moses.metrics import SNNMetric
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.property import tanimoto_similarity_pool as similarity_fcn
from Utils.property import get_mol, get_smiles, is_valid
from Inference.metrics import get_all_metrics, get_snn_from_mol, get_basic_metrics, print_all_metrics
from Utils.property import predict_props
from Utils.dataset import to_dataloader
from Model.build_model import get_model
from Inference.model_prediction import Predictor
from Inference.utils import prepare_generator


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

    def _get_fields(self):
        fields = [('smiles', self.SRC)]
        for i, c in enumerate(self.conditions):
            fields.append((f'{c}_s', self.COND[i]))
        for i, c in enumerate(self.conditions):
            fields.append((f'{c}_t', self.COND[i]))
        return fields

    def _prepare_dataset(self, smiles, props_t):
        with Pool(self.n_jobs) as pool:
            props_s = np.array(pool.map(predict_props, smiles))

        df_raw = pd.DataFrame({
            'smiles'  : smiles,
            'logP_s'  : props_s[:, 0],
            'tPSA_s'  : props_s[:, 1],
            'QED_s'   : props_s[:, 2],
            'logP_t'  : [props_t[0]]*len(smiles),
            'tPSA_t'  : [props_t[1]]*len(smiles),
            'QED_t'   : [props_t[2]]*len(smiles),
        })
        dataset = DataFrameDataset(df=df_raw, fields=self.fields)
        return df_raw, dataset

    def _rewrap_input(self, data):
        src = torch.LongTensor([[self.SRC.vocab.stoi[t]
                                 for t in data.smiles]])
        props_s, props_t = [], []
        for c in self.conditions:
            props_s.append(getattr(data, f'{c}_s'))
            props_t.append(getattr(data, f'{c}_t'))
        props_s = torch.Tensor([props_s])
        props_t = torch.Tensor([props_t])
        return src, props_s, props_t

    def _filter_valid_smiles(self, smiles):
        with Pool(self.n_jobs) as pool:
            valid = pool.map(is_valid, smiles)
        valid_id = [i for i in range(len(valid)) if i == 1]
        return [smiles[id] for id in valid_id]
    
    def _select_similar_smiles(self, src, gen):
        gen = self._filter_valid_smiles(gen)
        gen = [smi for smi in gen if smi != src]
        with Pool(self.n_jobs) as pool:
            res = pool.amap(similarity_fcn, gen, [src]*len(gen))
        similarity = res.get()
        sorted_gen = [val for (_, val) in sorted(zip(similarity, gen),
                      key=lambda x: x[0], reverse=True)]
        
        return sorted_gen[:np.ceil(self.n_samples/2.0)]

    def _select_smiles(self, src_smiles, gen_smiles):
        all_gen_smiles = []
        for i, src in enumerate(src_smiles):
            print('src:', src)
            print('gen:', gen_smiles[i])
            smi = self._select_similar_smiles(src, gen_smiles[i])
            print('smi:', smi)
            all_gen_smiles.extend(smi)
        return np.random.choice(all_gen_smiles, self.n_selections)
            
    def one_step_sampling(self, dataset):
        src_smiles, gen_smiles = [], []
        
        for i, data in enumerate(dataset):
            src, props_s_curr, props_t_curr = self._rewrap_input(data)
            z = self.generator.encode_smiles(src, props_s_curr)
            smiles, *_ = self.generator.sample_smiles(props_t_curr, z)
            src_smiles.append("".join(data.smiles))
            gen_smiles.append(smiles)
        return src_smiles, gen_smiles
        
    def multi_step_sampling(self, src_smiles, props_t, n_step, step_save_folder):
        props_s = np.array(predict_props(src_smiles))
        props_d = (np.array(props_t) - props_s) / n_step
        src_smiles = [src_smiles]

        df_raw, dataset = self._prepare_dataset(src_smiles*self.n_samples, props_t)        
        df_raw.to_csv(os.path.join(step_save_folder, "0.csv"))
        
        for i in range(n_step):
            print(f'# steps: {n_step}, curr step: {i+1}')
            src_smiles, gen_smiles = self.one_step_sampling(dataset)
            print('all gen smiles:\n', gen_smiles)
            src_smiles = self._select_smiles(src_smiles, gen_smiles)

            props_t = props_s + props_d * (i+1)
            df_raw, dataset = self._prepare_dataset(src_smiles, props_t)
            df_raw.to_csv(os.path.join(step_save_folder, f"{i+1}.csv"))

    def generate(self, src_smiles, props_t, save_folder):
        for n_step in self.n_steps:
            step_save_folder = os.path.join(save_folder, f"{n_step}_step")
            os.makedirs(step_save_folder, exist_ok=True)
            
            self.multi_step_sampling(src_smiles, props_t, n_step, step_save_folder)

            exit()
            
            
def fast_src_generation(args, toklen_data, train_smiles, 
                        scaler, SRC, TRG, COND, device, logger):

    for epoch in args.epoch_list:
        args.use_model_path = os.path.join(args.train_path,
                                           args.model_name,
                                           f'model_{epoch}.pt')
        generator = prepare_generator(args, SRC, TRG,
                                      toklen_data, scaler, device)

        save_folder = os.path.join(args.inference_path, 'src_generation', 
                                   args.model_name, str(epoch))
        os.makedirs(save_folder, exist_ok=True)
        
        LOG = logger(name='src generation',
                     log_path=os.path.join(save_folder, "records.log"))

        scgn = SrcGeneration(args, SRC, COND, generator, train_smiles, LOG, device)
        scgn.generate(args.src_smiles, args.trg_props, save_folder)

