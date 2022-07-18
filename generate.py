import os
import joblib
import argparse
import pandas as pd
import numpy as np
import pandas as pd
import glob

from rdkit import Chem, RDLogger
from moses.metrics import metrics

from Utils import allocate_gpu
from Utils.field import smiles_fields
from Utils.seed import set_seed
from Utils.property import property_prediction
from Configuration.config import options
from Inference.beam_search import BeamSearchTool
from Inference.model_prediction import ModelPrediction
from Model.build_model import build_mlpencoder, build_mlptransformer, build_transformer


def get_model(args, SRC, TRG, model_type):
    if model_type == 'transformer':
        model_path = 'molGCT/molgct.pt'
        return build_transformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model,
                                 args.d_ff, args.H, args.latent_dim, args.dropout, args.nconds,
                                 args.use_cond2dec, args.use_cond2lat, file_path=model_path)
    elif model_type == 'mlp_transformer':
        if args.epoch != 0:
            model_path = glob.glob(os.path.join(args.model_directory, 'best_*'))[0]
        else:
            model_path = None
        return build_mlptransformer(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model,
                                    args.d_ff, args.H, args.latent_dim, args.dropout, args.nconds,
                                    args.use_cond2dec, args.use_cond2lat, args.variational,
                                    args.molgct_path, file_path=model_path)
    elif model_type == 'mlp_encoder':
        if args.epoch != 0:
            model_path = glob.glob(os.path.join(args.model_directory, 'best_*'))[0]
        else:
            model_path = None
        return build_mlpencoder(len(SRC.vocab), len(TRG.vocab), args.N, args.d_model,
                                args.d_ff, args.H, args.latent_dim, args.dropout, args.nconds,
                                args.use_cond2dec, args.use_cond2lat, args.variational,
                                args.molgct_path, file_path=model_path)
    else:
        exit('ERROR - No Model Type:', model_type)


def generate_molecules(args, model, target_properties, TRG, scaler, toklen_data,
                      device, num_samplings=100, limit_samplings=500):
    def create_metrics_line(smiles, logp_diff, tpsa_diff, qed_diff, validity=None):
        # VALIDITY / UNIQUNESS / MAE / MSiE / MAX / MIN / SUCCESSFULNESS / INTDIV
        line = f'{len(np.unique(smiles)) / len(smiles) * 100:.3f}\t'\
               f'{logp_diff.apply(np.abs).mean():.3f}\t'\
               f'{tpsa_diff.apply(np.abs).mean():.3f}\t'\
               f'{qed_diff.apply(np.abs).mean():.3f}\t'\
               f'{logp_diff.mean():.3f}\t{tpsa_diff.mean():.3f}\t{qed_diff.mean():.3f}\t'\
               f'{logp_diff.max():.3f}\t{tpsa_diff.max():.3f}\t{qed_diff.max():.3f}\t'\
               f'{logp_diff.min():.3f}\t{tpsa_diff.min():.3f}\t{qed_diff.min():.3f}\t'\
               f'{100 if len(predictions) == num_samplings else 0:.3f}\t'\
               f'{metrics.internal_diversity(smiles):.3f}'
        if validity is not None:
            line = f'{validity * 100:.3f}\t' + line
        return line

    bsTool = BeamSearchTool(args.nconds, args.latent_dim,
                            args.max_strlen, model, args.use_cond2dec)
    predictor = ModelPrediction(getattr(model, args.decode_type), args.use_cond2dec)

    all_predictions = None

    for logp, tpsa, qed in target_properties:
        print(f"\n>>> Desirable Properties (logP, tPSA, QED): "\
              f"{logp:.2f}, {tpsa:.2f}, {qed:.2f}")
        
        generate_path = os.path.join(args.storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt')
        properties = np.array([[logp, tpsa, qed]])
        valid_smiles = []

        if args.decode_type == 'mlp_decode':
            noise = np.random.normal(0, 0.2, size=(1, args.nconds))
            properties = (properties-noise, properties)

        gen_mol, gen_smiles, validity = generate_molecules_from_properties(
                                            predictor, bsTool, properties, TRG,
                                            scaler, toklen_data, device, 
                                            num_samplings, limit_samplings
                                            )

        with open(generate_path, 'w', buffering=5) as sample_file:
            sample_file.write(
                "logp_t\ttpsa_t\tqed_t\tlogp_p\ttpsa_p\tqed_p\tsmiles\n")

            for i in range(len(gen_mol)):
                logp_p, tpsa_p, qed_p = (property_prediction[c](gen_mol[i])
                                            for c in args.conditions)
                line = f"{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t"\
                       f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}\t"\
                       f"{gen_smiles[i]}"
                sample_file.write(line+'\n')
                valid_smiles.append(gen_smiles[i])
                print(f"SMILES {i+1}:", line)

        metrics_path = os.path.join(args.storage_path, 
                       f'mean_{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt')
        # if os.path.exists(metrics_path):
        #     continue
        
        predictions = pd.read_csv(generate_path, sep='\t')
        
        predictions['logp_diff'] = predictions['logp_p'] - predictions['logp_t']
        predictions['tpsa_diff'] = predictions['tpsa_p'] - predictions['tpsa_t']
        predictions['qed_diff'] = predictions['qed_p'] - predictions['qed_t']

        with open(metrics_path, 'w') as mean_file_path:
            mean_file_path.write('validity\tuniqueness\t'\
                                 'logp_mae\ttpsa_mae\tqed_mae\t'\
                                 'logp_mse\ttpsa_mse\tqed_mse\t'\
                                 'logp_max\ttpsa_max\tqed_max\t'\
                                 'logp_min\ttpsa_min\tqed_min\t'\
                                 'successfulness\tdiversity\n')
            line = create_metrics_line(predictions['smiles'],
                                       predictions['logp_diff'],
                                       predictions['tpsa_diff'],
                                       predictions['qed_diff'],
                                       validity)
            mean_file_path.write(line+'\n')

        all_predictions = pd.concat([all_predictions, predictions], axis=0)

    with open(os.path.join(args.storage_path, 'output.txt'), 'w') as prediction_writer:
        all_predictions.reset_index(inplace=True)
        prediction_writer.write('uniqueness\t'\
                                'logp_mae\ttpsa_mae\tqed_mae\t'\
                                'logp_mse\ttpsa_mse\tqed_mse\t'\
                                'logp_max\ttpsa_max\tqed_max\t'\
                                'logp_min\ttpsa_min\tqed_min\t'\
                                'successfulness\tdiversity\n')
        prediction_writer.write(create_metrics_line(all_predictions['smiles'],
                                                    all_predictions['logp_diff'],
                                                    all_predictions['tpsa_diff'],
                                                    all_predictions['qed_diff'])+'\n')


def generate_molecules_from_properties(predictor, bsTool, conditions, TRG, scaler,
                                       toklen_data, device, num_samplings, sampling_limit):
    num_trials = 0
    valid_mol, valid_smiles = [], []
    
    while len(valid_smiles) < sampling_limit:
        num_trials += 1
        smiles, _, _ = bsTool.sample_molecule(conditions, toklen_data, predictor,
                                              TRG, scaler, device)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_mol.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol)) # convert to canonical smiles
        if len(valid_smiles) == num_samplings:
            break
    return valid_mol, valid_smiles, len(valid_smiles) / num_trials
    

def generate_demo(args, model, TRG, scaler, toklen_data, num_samplings, device):
    
    print(f'Enter desirable properties:'
          f'logP ({args.logp_lb} ~ {args.logp_ub}), '
          f'tPSA ({args.tpsa_lb} ~ {args.tpsa_ub}), '
          f'QED ({args.qed_lb} ~ {args.qed_ub})')
    
    while True:
        if_continue = input('Continue? Input Y/N: ')
        if if_continue == 'Y':
            logp = float(input('Enter logP: '))
            tpsa = float(input('Enter tPSA: '))
            qed = float(input('Enter QED: '))
            break
        elif if_continue == 'N':
            exit('Bye~ Bye~')
        else:
            print('Please enter Y/N !!!!')

    conditions = np.array([[logp, tpsa, qed]])

    if args.decode_type == 'mlp_decode':
        noise = np.random.normal(0, 0.2, size=(1, args.nconds))
        conditions = (conditions-noise, conditions)

    bsTool = BeamSearchTool(args.nconds, args.latent_dim,
                            args.max_strlen, model, args.use_cond2dec)
    predictor = ModelPrediction(
        getattr(model, args.decode_type), args.use_cond2dec)

    for i in range(num_samplings):
        smiles, _, _ = bsTool.sample_molecule(conditions, toklen_data,
                                              predictor, TRG, scaler, device)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            print(f'SMILES {i:<4}{smiles:<55} ->\t'
                  f'logP: {property_prediction[args.conditions[0]](mol):.2f}\t'
                  f'tPSA: {property_prediction[args.conditions[1]](mol):.2f}\t'
                  f'QED:  {property_prediction[args.conditions[2]](mol):.2f}')


if __name__ == "__main__":
    set_seed(seed=0)
    RDLogger.DisableLog('rdApp.*')

    parser = argparse.ArgumentParser()
    parser = options(parser)
    args = parser.parse_args()

    device = allocate_gpu()
    scaler = joblib.load(args.scaler_path)
    SRC, TRG = smiles_fields(smiles_field_path=args.field_path)
    toklen_data = pd.read_csv(args.toklen_path)

    model = get_model(args, SRC, TRG, model_type='mlp_encoder').to(device)
    model.eval()

    if args.demo:
        generate_demo(args, model, TRG, scaler, toklen_data,
                      num_samplings=20, device=device)
    else:
        os.makedirs(args.storage_path, exist_ok=True)
        num_points = 5
        target_properties = np.array(np.meshgrid(np.linspace(args.logp_lb, args.logp_ub, num=num_points),\
                                                 np.linspace(args.tpsa_lb, args.tpsa_ub, num=num_points),\
                                                 np.linspace(args.qed_lb, args.qed_ub, num=num_points)))\
                                                 .T.reshape(-1,3)
        generate_molecules(args, model, target_properties, TRG, scaler, toklen_data,
                           device, num_samplings=100, limit_samplings=500)

        # metrics_list = []
        # for logp, tpsa, qed in target_properties:
        #     metrics_list.append(pd.read_csv(os.path.join(args.storage_path,
        #                         f'mean_{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t'))
        # all_metrics = pd.concat(metrics_list).reset_index()
        # all_metrics.to_csv(os.path.join(args.storage_path, 'output.csv'))
