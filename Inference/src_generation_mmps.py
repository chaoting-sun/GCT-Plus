import os
import numpy as np
import pandas as pd
import torch
from time import time
from datetime import timedelta
from torchtext import data
from torchtext.data import Example, Dataset
from pathos.multiprocessing import ProcessingPool as Pool

from Inference.utils import prepare_generator
# from Utils.properties import tanimoto_similarity as similarity_fcn
from Utils.properties import MurckoScaffoldSimilarity as similarity_fcn
from Utils.properties import is_valid, predict_properties, get_mol_from_smiles
from Utils.dataset import get_dataset, get_loader
from Utils.field import untokenize
from Utils.metric import get_error


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
            res = pool.amap(similarity_fcn, smiles_list1, smiles_list2)
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


def get_similarity(similarity_fcn, smiles_list1, smiles_list2, n_jobs):
    with Pool(n_jobs) as pool:
        res = pool.amap(similarity_fcn, smiles_list1, smiles_list2)
    similarity_list = res.get()
    return similarity_list


def mse_fcn(delval):
    return sum(delval) / len(delval)


def mae_fcn(delval):
    abs_delval = [abs(v) for v in delval]
    return sum(abs_delval) / len(abs_delval)


def decode_smiles(z, prop, src_smi, trg_prop, generator, n_samples,
                  n_samples_each_time, property_list, n_jobs, LOG):
    
    all_valid_smis, all_unique_smis, all_high_sims = [], [], []
    dprop = { p: [] for p in property_list }
    mse = { p: 0 for p in property_list }
    mae = { p: 0 for p in property_list }
    amsd = { p: 0 for p in property_list }
    aard = { p: 0 for p in property_list }

    for sample_no in range(n_samples // n_samples_each_time):
        sid = n_samples_each_time * sample_no
        eid = n_samples_each_time * (sample_no+1)
        
        LOG.info('sample smiles...')
        gen_list, *_ = generator.sample_smiles(prop[sid:eid, :],
                                               z[sid:eid, :],
                                               transform=False
                                               )

        LOG.info('process smiles...')
        valid_smis, valid_mols = get_mol_from_smiles(gen_list, n_jobs)
        unique_smis = list(set(valid_smis))

        all_valid_smis.extend(valid_smis)
        all_unique_smis.extend(unique_smis)
        
        gen_prop = predict_properties(valid_mols, property_list, n_jobs)
        
        for p_no, p_name in enumerate(property_list):
            dp = gen_prop[p_name] - trg_prop[p_no]
            dprop[p_name].extend(dp.tolist())
            
            mse[p_name] = mse_fcn(dprop[p_name])
            mae[p_name] = mae_fcn(dprop[p_name])
            amsd[p_name] = (dp / trg_prop[p_no]).mean()
            aard[p_name] = (dp / trg_prop[p_no]).abs().mean()
                    
        src_smis = [src_smi]*len(unique_smis)
        sims = get_similarity(similarity_fcn, src_smis, unique_smis, n_jobs)
        
        for j, sim in enumerate(sims):
            if sim >= 0.5:
                all_high_sims.append(unique_smis[j])
        
        LOG.info(f'({n_samples_each_time*(sample_no+1)}) '
                 f'#valid: {len(all_valid_smis)}\t'
                 f'#unique: {len(all_unique_smis)}\t'
                 f'#high sim: {len(all_high_sims)}'
                )
        LOG.info("mse: %s", mse)
        LOG.info("mae: %s", mae)
        LOG.info("amsd: %s", amsd)
        LOG.info("aard: %s", aard)
        
    return (all_valid_smis, all_unique_smis, all_high_sims), (mse, mae, amsd, aard)


def mae(trg_prop, gen_prop):
    abs_dprop = np.zeros((len(trg_prop),))
    for i in range(len(trg_prop)):
        abs_dprop[i] = abs(gen_prop[i] - trg_prop[i])
    return abs_dprop.sum() / len(abs_dprop)


def mse(trg_prop, gen_prop):
    dprop = np.zeros((len(trg_prop),))
    for i in range(len(trg_prop)):
        dprop[i] = gen_prop[i] - trg_prop[i]
    return dprop.sum() / len(dprop)



def fast_src_generation_mmps(args, toklen_data, train_smiles, scaler,
                             SRC, TRG, PROP, device, logger): 
    elapsed_time = -time()

    for epoch in args.epoch_list:
        main_save_folder = os.path.join(args.inference_path,
                                        'fast_src_generation_mmps',
                                        args.benchmark,
                                        args.model_name,
                                        str(epoch))
        os.makedirs(main_save_folder, exist_ok=True)

        args.model_path = os.path.join(args.train_path,
                                       args.benchmark,
                                       args.model_name,
                                       f'model_{epoch}.pt')
        
        generator = prepare_generator(args, SRC, TRG, toklen_data, scaler, device)

        save_folder = os.path.join(args.inference_path, 'src_generation', args.benchmark,
                                    args.model_name, str(epoch))
        os.makedirs(save_folder, exist_ok=True)

        LOG = logger('src generation', os.path.join(save_folder, "records.log"))
        LOG.info(args)
        
        fields = get_fields(SRC, PROP, args.property_list)
        dataset = get_dataset(os.path.join(args.data_folder, 'prepared'), fields,
                              [None, None, args.data_name])[0]

        data_iter = get_iterator(dataset, batch_size=1)
        dataloader = get_loader(data_iter=data_iter,
                                property_list=args.property_list,
                                pad_id=SRC.vocab.stoi['<pad>'],
                                max_strlen=args.max_strlen,
                                device=device
                                )
        
        results = { 'src': [] }
        results.update({ f'src_{p}': [] for p in args.property_list })
        results.update({ f'trg_{p}': [] for p in args.property_list })
        results.update({ 'n_valid': [], 'n_unique': [], 'n_highsim': [] })
        for p in args.property_list:
            results[f'{p}_mse'] = []
            results[f'{p}_mae'] = []
            results[f'{p}_amsd'] = []
            results[f'{p}_aard'] = []

        for d, batch in enumerate(dataloader):
            if d == 100:
                break

            LOG.info(f'encode smiles {d}...')
            
            src_smi = untokenize(batch.src[0], SRC.vocab)
            src_prop = scaler.inverse_transform(batch.econds.cpu())[0]        
            trg_prop = scaler.inverse_transform(batch.dconds.cpu())[0]

            _, mu, logvar = generator.encode_smiles(batch.src, batch.econds, transform=False)
            std = torch.exp(0.5*logvar)
            
            LOG.info('augment latent space...')

            zs = torch.tile(mu, (args.n_samples, 1, 1))
            for i in range(args.n_samples):
                eps = torch.empty_like(mu).normal_(0, 0.4)
                zs[i, :, :] = eps.mul(std).add(mu)
            props = torch.tile(batch.dconds, (args.n_samples, 1))

            LOG.info('decode smiles...')

            smi_outs, errors = decode_smiles(zs,
                                             props,
                                             src_smi,
                                             trg_prop,
                                             generator,
                                             args.n_samples,
                                             args.n_samples_each_time,
                                             args.property_list,
                                             args.n_jobs,
                                             LOG
                                             )
            valid_smis, unique_smis, highsim_smi = smi_outs
            mse, mae, amsd, aard = errors

            results['src'].append(src_smi)
            results['n_valid'].append(len(valid_smis))
            results['n_unique'].append(len(unique_smis))
            results['n_highsim'].append(len(highsim_smi))

            for i in range(len(args.property_list)):
                results[f'src_{p}'].append(src_prop[i])
                results[f'trg_{p}'].append(trg_prop[i])

            for p in args.property_list:
                results[f'{p}_mse'].append(mse[p])
                results[f'{p}_mae'].append(mae[p])
                results[f'{p}_amsd'].append(amsd[p])
                results[f'{p}_aard'].append(aard[p])

            print('elapsed time:', str(timedelta(seconds=elapsed_time+time())))

            if (d+1) % 10 == 0:
                res = pd.DataFrame(results)
                res.to_csv(os.path.join(main_save_folder, 'results.csv'))

        print('results:\n', results)
