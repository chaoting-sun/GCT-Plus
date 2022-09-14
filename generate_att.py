import os
from re import L
from time import time
import joblib
import argparse
import pandas as pd
import numpy as np
import pandas as pd
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from moses.metrics import metrics
from torchtext import data

from Utils import allocate_gpu, get_fields
from Utils.mapper import mapper
from Utils.field import smiles_fields
from Utils.seed import set_seed
from Utils.property import property_prediction, get_mol
from Utils.dataset import to_dataloader
from Configuration.config import options
from Inference.beam_search import BeamSearchTool
from Inference.model_prediction import ModelPrediction
from Model.modules import create_source_mask
from Model.att_encoder import decode

# from Model.build_model import build_model

from Model.build_model import build_model


def _valid(valid_preds, preds): return len(valid_preds) / \
    len(preds)*100 if len(preds) else 0


def _unique(smiles): return len(np.unique(smiles)) / \
    len(smiles)*100 if len(smiles) > 0 else 0


def _novelty(gen_smiles, train, n_jobs):
    if len(gen_smiles) == 0:
        return 0
    close_pool = False
    if n_jobs != 1:
        pool = Pool(n_jobs)
        close_pool = True
    else:
        pool = 1
    mols = mapper(pool)(get_mol, gen_smiles)
    gen_novelty = metrics.novelty(mols, train, n_jobs)
    if close_pool:
        pool.close()
        pool.join()
    return gen_novelty*100


def _intdiv(valid_smiles): return metrics.internal_diversity(
    valid_smiles)*100 if len(valid_smiles) > 0 else 0


def _mae(valid_dif): return valid_dif.apply(np.abs).mean()
def _mse(valid_dif): return valid_dif.mean()
def _max(valid_dif): return valid_dif.max()
def _min(valid_dif): return valid_dif.min()


def generate_metrics_line(preds, train, n_jobs):
    valid_preds = preds.loc[preds['valid'] == 1].copy()
    valid_preds['logp_diff'] = valid_preds['logp_p'] - valid_preds['logp_t']
    valid_preds['tpsa_diff'] = valid_preds['tpsa_p'] - valid_preds['tpsa_t']
    valid_preds['qed_diff'] = valid_preds['qed_p'] - valid_preds['qed_t']

    header = 'valid(%)\tunique(%)\tnovelty(%)\tdiversity(%)\t'  \
             'logp_mae\ttpsa_mae\tqed_mae\t'                    \
             'logp_mse\ttpsa_mse\tqed_mse\t'                    \
             'logp_max\ttpsa_max\tqed_max\t'                    \
             'logp_min\ttpsa_min\tqed_min\t'                    \
             'logp_aard(%)\ttpsa_aard(%)\tqed_aard(%)\t'        \
             'logp_amsd(%)\ttpsa_amsd(%)\tqed_amsd(%)'          \

    line = f"{_valid(valid_preds, preds):.3f}\t"                     \
           f"{_unique(valid_preds['smiles']):.3f}\t"                 \
           f"{_novelty(valid_preds['smiles'], train, n_jobs):.3f}\t" \
           f"{_intdiv(valid_preds['smiles']):.3f}\t" \
           f"{_mae(valid_preds['logp_diff']):.3f}\t" \
           f"{_mae(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_mae(valid_preds['qed_diff']):.3f}\t"  \
           f"{_mse(valid_preds['logp_diff']):.3f}\t" \
           f"{_mse(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_mse(valid_preds['qed_diff']):.3f}\t"  \
           f"{_max(valid_preds['logp_diff']):.3f}\t" \
           f"{_max(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_max(valid_preds['qed_diff']):.3f}\t"  \
           f"{_min(valid_preds['logp_diff']):.3f}\t" \
           f"{_min(valid_preds['tpsa_diff']):.3f}\t" \
           f"{_min(valid_preds['qed_diff']):.3f}\t"  \
           f"{(valid_preds['logp_diff'] / valid_preds['logp_t']).abs().mean()*100:.3f}\t" \
           f"{(valid_preds['tpsa_diff'] / valid_preds['tpsa_t']).abs().mean()*100:.3f}\t" \
           f"{(valid_preds['qed_diff'] / valid_preds['qed_t']).abs().mean()*100:.3f}\t"   \
           f"{(valid_preds['logp_diff'] / valid_preds['logp_t']).mean()*100:.3f}\t"       \
           f"{(valid_preds['tpsa_diff'] / valid_preds['tpsa_t']).mean()*100:.3f}\t"       \
           f"{(valid_preds['qed_diff'] / valid_preds['qed_t']).mean()*100:.3f}"           \

    return header, line


def get_model(args, SRC_vocab_len, TRG_vocab_len, model_type, decode_type):
    if model_type == 'transformer':
        model_path = 'molGCT/molgct.pt'
    elif args.epoch > 0:
        model_path = os.path.join(args.model_directory, f'model_{args.epoch}.pt')
        # model_path = glob.glob(os.path.join(args.model_directory, 'best_*'))[0]
    else:
        model_path = None
    print("Model path:", model_path)
    return build_model(args, SRC_vocab_len, TRG_vocab_len, model_path)


# def predict_molecules(args, bsTool, predictor, properties, TRG, scaler,
#                       toklen_data, device, smiles_path, num_samplings=500):
#     logp, tpsa, qed = properties
    
#     print(f"\nDesirable Properties (logP, tPSA, QED): "
#           f"{logp:.2f}, {tpsa:.2f}, {qed:.2f}")
    
#     properties = np.array([[logp, tpsa, qed]])

#     if args.decode_type == 'mlp_decode':
#         noise = np.random.normal(0, 0.2, size=(1, args.nconds))
#         properties = (properties-noise, properties)
#     generated_smiles = generate_smiles_from_properties(predictor, bsTool, properties, TRG,
#                                                        scaler, toklen_data, device, num_samplings)

#     with open(smiles_path, 'w', buffering=10) as sample_file:
#         sample_file.write(f"number\tsmiles\tvalid\t"
#                           f"logp_t\ttpsa_t\tqed_t\t"
#                           f"logp_p\ttpsa_p\tqed_p\n")

#         for i in range(num_samplings):
#             mol = Chem.MolFromSmiles(generated_smiles[i])
#             logp_p = tpsa_p = qed_p = np.nan
#             valid = 0
#             if mol is not None:
#                 valid = 1
#                 logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
#                                             for c in args.conditions)
#             line = f"{i+1}\t{generated_smiles[i]}\t{valid}\t"  \
#                    f"{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t"      \
#                    f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}"

#             sample_file.write(line+'\n')
#             print(f"SMILES {i+1}:", line)


def generate_smiles_from_properties(args, bsTool, predictor, properties, TRG, scaler,
                                    toklen_data, device, num_samplings=500, mu=0, std=0.2):
    logp, tpsa, qed = properties
    
    properties = np.array([[logp, tpsa, qed]])

    if args.decode_type == 'mlp_decode':
        noise = np.random.normal(mu, std, size=(1, args.nconds))
        properties = (properties-noise, properties)

    return [bsTool.sample_molecule(properties, toklen_data,
            predictor, TRG, scaler, device)[0] for _ in range(num_samplings)]


def store_properties_from_predicted_smiles(args, properties, property_prediction, 
                                           generated_smiles, smiles_path):
    logp, tpsa, qed = properties

    with open(smiles_path, 'w', buffering=10) as sample_file:
        sample_file.write(f"number\tsmiles\tvalid\t"
                          f"logp_t\ttpsa_t\tqed_t\t"
                          f"logp_p\ttpsa_p\tqed_p\n")

        for i in range(len(generated_smiles)):
            print(f'{i:>5} Compute properties:', generated_smiles[i])
            mol = Chem.MolFromSmiles(generated_smiles[i])
            if mol is not None:
                valid = 1
                logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
                                         for c in args.conditions)
            else:
                valid = 0
                logp_p = tpsa_p = qed_p = np.nan
            line = f"{i+1}\t{generated_smiles[i]}\t{valid}\t"  \
                   f"{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t"      \
                   f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}"

            sample_file.write(line+'\n')
            print(f'- {logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}')


def generate_demo(args, model, logp, tpsa, qed, TRG, scaler,
                  toklen_data, num_samplings, device, mu=0, std=0.2):
    conditions = np.array([[logp, tpsa, qed]])

    if args.decode_type == 'mlp_decode':
        noise = np.random.normal(mu, std, size=(1, args.nconds))
        conditions = (conditions-noise, conditions)

    bsTool = BeamSearchTool(args.nconds, args.latent_dim,
                            args.max_strlen, model, args.use_cond2dec)
    predictor = ModelPrediction(getattr(model, args.decode_type), args.use_cond2dec)

    for i in range(num_samplings):
        smiles, _, _ = bsTool.sample_molecule(conditions, toklen_data,
                                              predictor, TRG, scaler, device)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            print(f'SMILES {i+1:<6}{smiles:<55} ->\t'
                  f'logP: {property_prediction[args.conditions[0]](mol):.2f}\t'
                  f'tPSA: {property_prediction[args.conditions[1]](mol):.2f}\t'
                  f'QED:  {property_prediction[args.conditions[2]](mol):.2f}')


def sample_source_from_filepath(infile_path, oufile_path, n_samples, state=0):
    df = pd.read_csv(infile_path)
    df = df.sample(n=n_samples, random_state=state)
    df.to_csv(oufile_path)


if __name__ == "__main__":
    # set_seed(seed=0)
    RDLogger.DisableLog('rdApp.*')

    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    print('-------------------------- Settings --------------------------')
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    print('-------------------------- Settings --------------------------')

    device = allocate_gpu()
    scaler = joblib.load(args.scaler_path)
    fields, SRC, TRG = get_fields(args.conditions, args.field_path)
    toklen_data = pd.read_csv(args.toklen_path)

    print(TRG.vocab.stoi)

    sos_idx = TRG.vocab.stoi['<sos>']
    eos_idx = TRG.vocab.stoi['<eos>']
    pad_idx = SRC.vocab.stoi['<pad>']

    n_samples = 100
    data_type = 'train'

    source_data_folder = os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}')
    sample_data_folder = os.path.join(source_data_folder, 'phaseI-inference')
    os.makedirs(sample_data_folder, exist_ok=True)
    
    sample_source_from_filepath(infile_path=os.path.join(source_data_folder, data_type+'.csv'),
                                oufile_path=os.path.join(sample_data_folder, data_type+'.csv'),
                                n_samples=n_samples, state=0)

    dataset = data.TabularDataset(path=os.path.join(sample_data_folder, data_type+'.csv'),
                                  format='csv', fields=fields, skip_header=True)
    data_iter = data.BucketIterator(dataset, batch_size=1)

    model = get_model(args, len(SRC.vocab), len(TRG.vocab), 
                      args.model_type, args.decode_type).to(device)    
    model.eval()

    dataloader = to_dataloader(data_iter, args.conditions, pad_idx, args.max_strlen, device)

    for i, batch in enumerate(dataloader):
        # src_pad_mask = create_source_mask(batch.src, pad_idx, batch.econds)
        # model.encode_att_sample(batch.src, batch.econds, batch.mconds, src_pad_mask)
        smi_sequence = decode(model, batch.src, batch.econds, batch.mconds, batch.dconds,
                              sos_idx, eos_idx, pad_idx, args.max_strlen, 'multinomial',
                              args.use_cond2dec)
        print(smi_sequence)
        # break
    
    exit(0)

    target_properties = np.array(np.meshgrid(np.linspace(args.logp_lb, args.logp_ub, num=args.num_points),
                                             np.linspace(args.tpsa_lb, args.tpsa_ub, num=args.num_points),
                                             np.linspace(args.qed_lb, args.qed_ub, num=args.num_points))) \
        .T.reshape(-1, 3)

    generate_smiles_time = store_properties_time = 0
    os.makedirs(args.storage_path, exist_ok=True)

    total_time = time()

    bsTool = BeamSearchTool(args.nconds, args.latent_dim,
                            args.max_strlen, model, args.use_cond2dec)
    predictor = ModelPrediction(getattr(model, args.decode_type), args.use_cond2dec)

    for i, (logp, tpsa, qed) in enumerate(target_properties):    
        properties = (logp, tpsa, qed)

        print(f"\n({i}) Desirable Properties (logP, tPSA, QED): "
                f"{logp:.2f}, {tpsa:.2f}, {qed:.2f}")

        smiles_path = os.path.join(args.storage_path,
                        f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_pre.txt')
        smiles_property_path = os.path.join(args.storage_path,
                                f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt')

        if os.path.exists(smiles_property_path):
            continue
        
        generate_smiles_time -= time()
        generated_smiles = generate_smiles_from_properties(args, bsTool, predictor, properties, TRG,
                                                            scaler, toklen_data, device, args.samples_each)
        generate_smiles_time += time()

        with open(smiles_path, 'w') as sample_file:
            sample_file.write(f"number\tsmiles\n")
            for i in range(len(generated_smiles)):
                sample_file.write(f"{i+1}\t{generated_smiles[i]}\n")            

        store_properties_time -= time()
        store_properties_from_predicted_smiles(args, properties, property_prediction, 
                                                generated_smiles, smiles_property_path)
        store_properties_time += time()

        print(f"generateT(s): {generate_smiles_time}\t"
                f"storepropT(s): {store_properties_time}\t"
                f"totalT(s): {time() - total_time}")
        os.remove(smiles_path)

    print("Compute metrics for generated smiles of each desirable properties")
    
    for logp, tpsa, qed in target_properties:
        mean_file_path = os.path.join(args.storage_path, 
                            f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt')
        print(">>>", mean_file_path)
        with open(mean_file_path, 'w') as metrics_writer:
            preds = pd.read_csv(os.path.join(args.storage_path,
                    f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
            head, line = generate_metrics_line(preds, train_smiles, args.n_jobs)
            metrics_writer.write('logp\ttpsa\tqed\t'+head+'\n')
            metrics_writer.write(f'{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t'+line+'\n')

    print("Combine all of the metrics computed before.")

    all_metrics = None
    
    for i, (logp, tpsa, qed) in enumerate(target_properties):
        preds = pd.read_csv(os.path.join(args.storage_path, 
                f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt'), sep='\t')
        all_metrics = pd.concat([all_metrics, preds], axis=0, ignore_index=True)

    # all_metrics = pd.concat([properties, all_metrics], axis=1)
    # print(all_metrics.head())
    all_metrics = all_metrics.sort_values(by=['logp', 'tpsa', 'qed'])
    all_metrics.to_csv(os.path.join(args.storage_path, 'mean.txt'), sep='\t', index=False)
    
    print("Compute metrics for smiles of all property combinations")

    # if not os.path.exists(os.path.join(args.storage_path, 'output.txt')):
    all_preds = None
    for logp, tpsa, qed in target_properties:
        preds = pd.read_csv(os.path.join(args.storage_path,
                            f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
        all_preds = pd.concat([all_preds, preds], axis=0)
    all_preds = all_preds.reset_index()

    with open(os.path.join(args.storage_path, 'output.txt'), 'w') as all_metrics_writer:
        head, line = generate_metrics_line(all_preds, train_smiles, args.n_jobs)
        all_metrics_writer.write(head+'\n')
        all_metrics_writer.write(line+'\n')

    print("Compute metrics for smiles of all property combinations except for logP=0.03")

    # if not os.path.exists(os.path.join(args.storage_path, 'output-logp0.03.txt')):
    all_preds = None
    for logp, tpsa, qed in target_properties:
        if logp == 0.03:
            continue
        preds = pd.read_csv(os.path.join(args.storage_path,
                            f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
        all_preds = pd.concat([all_preds, preds], axis=0)
    all_preds = all_preds.reset_index()

    with open(os.path.join(args.storage_path, 'output-logp0.03.txt'), 'w') as all_metrics_writer:
        head, line = generate_metrics_line(all_preds, train_smiles, args.n_jobs)
        all_metrics_writer.write(head+'\n')
        all_metrics_writer.write(line+'\n')
    
    print("Work Finished")
