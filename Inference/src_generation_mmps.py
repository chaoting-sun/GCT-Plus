import os
import numpy as np
import pandas as pd
import torch
from time import time
from datetime import timedelta
from torchtext import data
from torchtext.data import Example, Dataset
from pathos.multiprocessing import ProcessingPool as Pool
from typing import Callable, List
from collections import OrderedDict

from Inference.utils import prepare_generator
# from Utils.properties import tanimoto_similarity as similarity_fn
from Utils.properties import MurckoScaffoldSimilarity as similarity_fn
# from Utils.properties import predict_properties
from Utils.smiles import is_valid
from Utils.dataset import get_dataset, get_loader
from Utils.field import untokenize

from Utils.dataset import SmilesDataset
from Utils.dataset import DataloaderPreparation

# train

# 0.50: 62508693
# 0.60: 
# 0.70: 7111342
# 0.75: 3989198
# 0.80: 2354581


class DataFrameDataset(Dataset):
    def __init__(self, df, SRC, PROP, property_list):
        # get fields for dataset
        fields = [('smiles', SRC)]
        for i, c in enumerate(property_list):
            fields.append((f'src_{c}', PROP[i]))
        for i, c in enumerate(property_list):
            fields.append((f'trg_{c}', PROP[i]))
        
        super(DataFrameDataset, self).__init__([
                Example.fromlist(list(r), fields)
                for i, r in df.iterrows()
            ], fields
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
            res = pool.amap(similarity_fn, smiles_list1, smiles_list2)
        similarity_list = res.get()
        return similarity_list


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
            zs = self.augment_latent_space(mu, std, ep_std=0) # variable: ep_std

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


def get_fields(SRC, PROP, property_list):
    fields = [('src', SRC)]
    for i, p in enumerate(property_list):
        fields.append((f'src_{p}', PROP[i]))
    fields.append(('trg', None))
    for i, p in enumerate(property_list):
        fields.append((f'trg_{p}', PROP[i]))
    return fields


def get_iterator(dataset, batch_size):
    data_iter = data.BucketIterator(
        dataset, batch_size, shuffle=False
    )
    return data_iter


def get_similarity(similarity_fn, smiles_list1, smiles_list2, n_jobs):
    with Pool(n_jobs) as pool:
        res = pool.amap(similarity_fn, smiles_list1, smiles_list2)
    similarity_list = res.get()
    return similarity_list


def MSE(trgv, genv):
    delv = [genv[i] - trgv[i] for i in range(len(trgv))]
    return sum(delv) / len(delv)


def MAE(trgv, genv):
    abs_delv = [abs(genv[i] - trgv[i]) for i in range(len(trgv))]
    return sum(abs_delv) / len(abs_delv)
    
    
def AMSD(trgv, genv):
    delv = [genv[i] - trgv[i] for i in range(len(trgv))]
    rel_delv = [delv[i] / trgv[i] for i in range(len(trgv))]
    return sum(rel_delv) / len(rel_delv)


def AARD(trgv, genv):
    abs_delv = [abs(genv[i] - trgv[i]) for i in range(len(trgv))]
    rel_abs_delv = [abs_delv[i] / trgv[i] for i in range(len(trgv))]
    return sum(rel_abs_delv) / len(rel_abs_delv)


def compute_errors(
        trgv: np.array,
        genv: pd.DataFrame,
        desired_properties: List,
        error_fn: List[Callable]
    ):
    error_dict = OrderedDict()
    for i, p in enumerate(desired_properties):
        trgv_list = [trgv[i] for _ in range(len(genv))]
        genv_list = genv[p]
        error_dict.update((f'{fn.__name__}_{p}',
                           fn(trgv_list, genv_list))
                          for fn in error_fn)
    return error_dict


def compute_records(
        src_smi: str,
        trgv: np.array,
        gen_smi: List,
        desired_properties: List,
        error_fn: List[Callable],
        similarity_fn: Callable,
        n_jobs=1
    ):
    
    valid_smi, valid_mol = get_mol_from_smiles(gen_smi, n_jobs)
    unique_smi = list(set(valid_smi))
    
    genv = predict_properties(valid_mol, desired_properties, n_jobs)
    error_dict = compute_errors(trgv, genv, desired_properties, error_fn)
    
    smi_sim = get_similarity(similarity_fn, [src_smi]*len(unique_smi), 
                             unique_smi, n_jobs)
    
    highsim_unique_smi = []
    for i, sim in enumerate(smi_sim):
        if sim is not None and sim >= 0.5:
            highsim_unique_smi.append(unique_smi[i])

    return highsim_unique_smi, len(valid_smi), len(unique_smi), error_dict


def initialize_records(
        desired_properties, error_fn,
        recorded_nums=('n_valid', 'n_unique', 'n_highsim')
    ):
    records = { 'src': [] }
    
    records.update({ f'src_{p}': [] for p in desired_properties })
    records.update({ f'trg_{p}': [] for p in desired_properties })
    records.update({ nums: [] for nums in recorded_nums })
    for p in desired_properties:
        records.update({ f'{fn.__name__}_{p}': [] for fn in error_fn })
    return records


def augment_decoder_inputs(mu, logvar, prop, n,
                           ep_mu=0, ep_std=1):
                        #    ep_mu=0, ep_std=0.3):
                        #    ep_mu=0, ep_std=0.6):
    props = torch.tile(prop, (n, 1))
    std = torch.exp(0.5*logvar)
    
    """(1)"""
    # augment all
    stds = torch.tile(std, (n, 1, 1))
    mus = torch.tile(mu, (n, 1, 1))
    eps = torch.zeros((n, mu.size(1), mu.size(2)),
                      device=mu.device
                      ).normal_(ep_mu, ep_std)
    
    zs = eps.mul(stds).add(mus)
    """(2)"""
    # augment 1, 2, 3, 4, 5 positions
    # zs = torch.zeros((n, mu.size(1), mu.size(2)),
    #                  device=mu.device)
    # for i in range(len(mu)):
    #     pos_id = np.random.choice(range(mu.size(1)), 10)
    #     eps = torch.zeros((1, mu.size(1), mu.size(2)),
    #                       device=mu.device
    #                       ).normal_(ep_mu, ep_std)
    #     zs[i, :, pos_id] = eps.mul(std).add(mu)[0, :, pos_id]
        
    return zs, props


def fast_src_generation_mmps(args, toklen_data, train_smiles, scaler,
                             SRC, TRG, PROP, device, logger):    
    epoch = args.epoch_list[0]

    save_folder = os.path.join(args.inference_path, 'fast_src_generation_mmps',
                               args.benchmark, args.model_name, str(epoch))
    os.makedirs(save_folder, exist_ok=True)

    LOG = logger('fast_src_generation_mmps',
                 os.path.join(save_folder, "records.log"))
    LOG.info(args)

    print('src:', SRC.vocab.stoi)
    print('trg:', TRG.vocab.stoi)

    LOG.info('get generator...')

    args.model_path = os.path.join(
        args.train_path,
        args.benchmark,
        args.model_name,
        f'model_{epoch}.pt'
    )
        
    generator = prepare_generator(args, SRC, TRG, toklen_data, scaler, device)

    LOG.info('get dataloader...')
    
    file_path = os.path.join(args.data_folder, 'prepared', f'{args.data_name}.csv')
    
    dp = DataloaderPreparation(
        SRC, TRG, args.property_list, batch_size=1,
        is_train=False, rank=0, world_size=1
    )
    
    dataset = dp.get_dataset(file_path)
    dataloader = dp.get_dataloader(dataset)

    LOG.info('initialize records...')

    error_fn = [MSE, MAE, AMSD, AARD]
    
    records = initialize_records(args.property_list, error_fn)

    LOG.info('start sampling smiles...')

    np.random.seed(123)
    n_transform_choices = 100
    transform_choices = np.random.choice(np.arange(len(dataset)), 
                                         n_transform_choices)
    transform_choices.sort()

    no = 0
    for d, batch in enumerate(dataloader):
        if no < len(transform_choices) and d != transform_choices[no]:
            # if d is not selected in transform_choices, turn to the next one
            continue
        if d > transform_choices[-1]:
            # if d is larger than the last one of transform_choices
            # the job has been done
            break
        
        LOG.info(f'#{no} -> {d}')
        
        if args.model_type == 'cvaetf':
            prop = batch['econds']
        
        elif args.model_type == 'attencvaetf':
            prop = (batch['econds'], batch['mconds'])

        """cvaetf-s1.00, attencvaetf-mconds-s0.70"""
        
        if args.model_name == 'attencvaetf-z-s0.70':
            gen_smi = []
            src = torch.tile(batch['src'], (100, 1))
            econds = (torch.tile(prop[0], (100, 1)),
                    torch.tile(prop[1], (100, 1)))
            dconds = torch.tile(batch['dconds'], (100, 1))

            for i in range(args.n_samples // 100):
                zs, *_ = generator.encode_smiles(src, econds, transform=False)        
                smi, *_ = generator.sample_smiles(dconds, zs, transform=False)
                gen_smi.extend(smi)
        
        else:
            _, mu, logvar = generator.encode_smiles(batch['src'], prop, transform=False)
            zs, props = augment_decoder_inputs(mu, logvar, batch['dconds'], args.n_samples)

            gen_smi = []
            n_samples_each_time = 1000
            
            for i in range(args.n_samples // n_samples_each_time):
                start_id = i * n_samples_each_time
                end_id = (i+1) * n_samples_each_time
                smi, *_ = generator.sample_smiles(
                    props[start_id:end_id, :],
                    zs[start_id:end_id, :, :],
                    transform=False)
                gen_smi.extend(smi)
                


        # compute records
            
        src_smi = untokenize(batch['src'][0], SRC.vocab)
        srcv = scaler.inverse_transform(batch['econds'].cpu())[0]
        trgv = scaler.inverse_transform(batch['dconds'].cpu())[0]
        
        highsim_unique_smi, n_valid, n_unique, error_dict = compute_records(
            src_smi, trgv, gen_smi, args.property_list, error_fn,
            similarity_fn, args.n_jobs)
        
        LOG.info(f'soure smiles: {src_smi}')
        LOG.info(f'#valid: {n_valid}, #unique: {n_unique}, #highsim: {len(highsim_unique_smi)}')
        # LOG.info(records)
        
        # save records
        
        records['src'].append(src_smi)
        records['n_valid'].append(n_valid)
        records['n_unique'].append(n_unique)
        records['n_highsim'].append(len(highsim_unique_smi))

        for i, p in enumerate(args.property_list):
            records[f'src_{p}'].append(srcv[i])
            records[f'trg_{p}'].append(trgv[i])
            
            for fn in error_fn:
                name = f'{fn.__name__}_{p}'
                records[name].append(error_dict[name])

        if (no+1) % 10 == 0:
            res = pd.DataFrame(records)
            res.to_csv(os.path.join(save_folder, 'records.csv'))

        no += 1

    print('records:\n', records)
