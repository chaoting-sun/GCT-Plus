from plot import smilesList_to_img  # in SMILES_plot
import dill as pickle
import os
from re import L
from time import time
import torch
import joblib
import argparse
import pandas as pd
import numpy as np
import pandas as pd
from multiprocessing import Pool
from torchtext import data
from Utils import allocate_gpu, get_fields

from rdkit import Chem, RDLogger
from moses.metrics import metrics
from Utils.dataset import to_dataloader

from Utils import allocate_gpu
from Utils.mapper import mapper
from Utils.field import smiles_fields
from Utils.seed import set_seed
from Utils.property import property_prediction, get_mol, tanimoto_similarity
from Configuration.config import options
# from Inference.beam_search import BeamSearchTool
from Inference.demo import Demo
from Inference.model_prediction import Predictor
# from Model.build_model import build_model
from moses.metrics import SNNMetric

from Inference.decode_algo3 import MultinomialSearch, MultinomialSearchFromSource, BeamSearch, generate_latent_space
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
        model_path = os.path.join(args.molgct_path, 'molgct.pt')
    elif args.epoch > 0:
        model_path = os.path.join(
            args.model_directory, f'model_{args.epoch}.pt')
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


def store_properties_from_predicted_smiles(properties, property_prediction,
                                           generated_smiles, smiles_path, conditions):
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
                                         for c in conditions)
            else:
                valid = 0
                logp_p = tpsa_p = qed_p = np.nan
            line = f"{i+1}\t{generated_smiles[i]}\t{valid}\t"  \
                   f"{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t"      \
                   f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}"

            sample_file.write(line+'\n')
            print(f'- {logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}')


# def generate_smiles_from_src(args, fields, TRG):
#     test_file_path = "./test_smiles.csv"
#     target_smiles = 'CNC(=O)c1cccc(NCC(=O)Nc2cccc(C(=O)NC)c2)c1'

#     if not os.path.exists(test_file_path):
#         df = pd.read_csv(os.path.join(args.data_path, 'aug', f'data_sim{args.similarity:.2f}', 'train.csv'))
#         df = df.loc[df.src == target_smiles]
#         df.to_csv(test_file_path, index=False)

#     df = pd.read_csv(test_file_path)

#     dataset = data.TabularDataset(path=test_file_path, format='csv', fields=fields, skip_header=True)
#     data_iter = data.BucketIterator(dataset, batch_size=1)
#     dataloader = to_dataloader(data_iter, args.conditions, TRG.vocab.stoi['<pad>'], args.max_strlen, device)

#     predictor = Predictor(getattr(model, args.decode_type), args.use_cond2dec)

#     demo = Demo(args.conditions, predictor, args.decode_type, args.latent_dim,
#                 args.max_strlen, args.use_cond2dec, toklen_data, scaler, TRG,
#                 (args.logp_lb, args.logp_ub), (args.tpsa_lb, args.tpsa_ub),
#                 (args.qed_lb, args.qed_ub), 'beam_search', device)


#     for i, batch in enumerate(dataloader):
#         properties = batch.dconds.cpu().numpy()[0]
#         demo.inference_from_src_properties(batch.src, properties[0], properties[1], properties[2])


def calc_dist(z): return torch.sqrt(torch.sum(z**2)).item()


def calc_snn_from_mol(molList1, molList2):
    # Similarity to nearest neighbour
    # molList1_fp = SNNMetric().precalc()
    return SNNMetric()(gen=molList1, ref=molList2)


def smi_to_mol(smilesList):
    molList = []
    for s in smilesList:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            molList.append(mol)
    return molList


class VaryingZGeneration:
    def __init__(self, generator, latent_dim, distance, save_folder):
        # save_folder -> dist1
        self.generator = generator
        self.latent_dim = latent_dim
        self.distance = distance
        self.save_folder = save_folder
        self.zs_path = os.path.join(save_folder, "zs")

        os.makedirs(self.zs_path, exist_ok=True)

    def generate_rand_z(self, toklen, n=1):
        return torch.Tensor(np.random.normal(size=(n, toklen, self.latent_dim)))

    def generate_rand_scaled_z(self, toklen, start_z):
        new_z = self.generate_rand_z(toklen)
        del_new_z = new_z - start_z
        return start_z + del_new_z * self.distance / calc_dist(del_new_z)

    def get_z_same_dist_from_start_z(self, toklen, num_z, start_z):
        def take_z(file_path):
            if not os.path.exists(file_path):
                z = self.generate_rand_scaled_z(toklen, start_z)
                pickle.dump(z.cpu(), open(file_path, "wb"))
            else:
                z = pickle.load(open(file_path, "rb"))
            return z

        z_list = [
            take_z(os.path.join(self.zs_path, f"z{i+1}")) for i in range(num_z)]
        return [start_z] + z_list

    def sample_molecules_from_varying_z(self, properties, toklen, n_z, n_samples):
        logp, tpsa, qed = properties
        conditions = np.array([[logp, tpsa, qed]])

        save_path = os.path.join(
            self.save_folder, f"cond_{logp:.2f}_{tpsa:.2f}_{qed:.2f}")
        os.makedirs(save_path, exist_ok=True)

        start_z_path = os.path.join(self.zs_path, "start_z.pkl")
        if not os.path.exists(start_z_path):
            start_z = self.generate_rand_z(toklen)
            pickle.dump(start_z.cpu(), open(start_z_path, "wb"))
        else:
            start_z = pickle.load(open(start_z_path, "rb"))

        z_list = self.get_z_same_dist_from_start_z(toklen, n_z, start_z)

        snnList_start, snnList_prev = [], [0]
        molList_start, molList_prev, molList = [], [], []

        for i, z in enumerate(z_list):
            z = z.to(device)
            smilesList = [self.generator.sample_smiles(
                conditions, z)[0] for _ in range(n_samples)]

            print("----------- smiles list -----------\n", smilesList)
            with open(os.path.join(save_path, f"{i}.txt"), "w") as writer:
                for s in smilesList:
                    writer.write(s+"\n")

            if len(molList_start) == 0:
                molList_start = smi_to_mol(smilesList)
            molList = smi_to_mol(smilesList)

            snn_start = calc_snn_from_mol(molList_start, molList)
            snnList_start.append(snn_start)

            if len(molList_prev) != 0:
                snn_prev = calc_snn_from_mol(molList_prev, molList)
                snnList_prev.append(snn_prev)

            molList_prev = molList

        with open(os.path.join(save_path, f"snn.txt"), "w") as writer:
            writer.write("snn_start\tsnn_prev\n")
            for i in range(len(snnList_start)):
                writer.write(f"{snnList_start[i]}\t{snnList_prev[i]}\n")


def getSmilesGenerator(args, model, decode_algo, has_src):
    predictor = Predictor(getattr(model, args.decode_type), args.use_cond2dec)
    z_generator = model.encode

    if decode_algo == "multinomial":
        if has_src:
            smiles_generator = MultinomialSearch(
                predictor, args.latent_dim, TRG, toklen_data,
                scaler, args.max_strlen, args.use_cond2dec, device
            )
        else:
            smiles_generator = MultinomialSearchFromSource(
                z_generator, predictor, args.latent_dim, TRG, toklen_data,
                scaler, args.max_strlen, args.use_cond2dec, device
            )

    elif decode_algo == "beam_search":
        if has_src:
            smiles_generator = BeamSearch(
                predictor, args.latent_dim, TRG, toklen_data,
                scaler, args.max_strlen, args.use_cond2dec, device
            )
        else:
            smiles_generator = BeamSearch(
                predictor, args.latent_dim, TRG, toklen_data,
                scaler, args.max_strlen, args.use_cond2dec, device
            )
    return smiles_generator


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
    scaler = joblib.load(os.path.join(args.molgct_path, 'scaler.pkl'))
    fields, SRC, TRG = get_fields(args.conditions, args.molgct_path)

    toklen_data = pd.read_csv(os.path.join(
        args.data_path, 'raw', 'train', 'toklen_list.csv'))

    train_smiles = pd.read_csv(os.path.join(
        args.data_path, 'raw', 'train', 'smiles_serial.csv'))
    train_smiles = train_smiles['smiles'].tolist()

    model = get_model(args, len(SRC.vocab), len(TRG.vocab),
                      args.model_type, args.decode_type).to(device)
    model.eval()

    decode_algo = "multinomial"
    has_src = True

    n_z = 5
    toklen = 40
    n_samples = 20
    distance = [1, 4, 8, 16, 32]
    propertyList = [(2.22, 48.81, 0.84), (3.25, 89.95, 0.73),
                    (2.01, 42.51, 0.63)]

    smiles_generator = getSmilesGenerator(args, model, decode_algo, has_src)

    if has_src:
        save_folder = f"/fileserver-gamma/chaoting/ML/molGCT/propsrc-{decode_algo}/"
        dataset = data.TabularDataset(path=os.path.join(save_folder, 'test.csv'),
                                      format='csv', fields=fields, skip_header=True)
        data_iter = data.BucketIterator(dataset, batch_size=1)
        dataloader = to_dataloader(
            data_iter, args.conditions, TRG.vocab.stoi["<pad>"], args.max_strlen, device)
        batch = next(dataloader)

        smiles_generator.sample_smiles(batch.src, propertyList[0])
        exit()

        for dist in distance:
            save_folder = f"/fileserver-gamma/chaoting/ML/molGCT/propsrc-{decode_algo}/dist{dist}"
            obj = VaryingZGeneration(
                smiles_generator, args.latent_dim, dist, save_folder)

            for prop in propertyList:
                obj.sample_molecules_from_varying_z(
                    prop, toklen, n_z, n_samples)
    else:
        for dist in distance:
            save_folder = f"/fileserver-gamma/chaoting/ML/molGCT/prop-{decode_algo}/dist{dist}"
            obj = VaryingZGeneration(
                smiles_generator, args.latent_dim, dist, save_folder)

            for prop in propertyList:
                obj.sample_molecules_from_varying_z(
                    prop, toklen, n_z, n_samples)

    exit(0)

    if args.demo:
        print(f'Enter desirable properties:'
              f'logP ({args.logp_lb} - {args.logp_ub}), '
              f'tPSA ({args.tpsa_lb} - {args.tpsa_ub}), '
              f'QED ({args.qed_lb} - {args.qed_ub})')

        while True:
            # if_continue = input('Continue? Input Y/N: ')
            if_continue = 'Y'
            if if_continue.upper() == 'Y':
                # logp = float(input('Enter logP: '))
                # tpsa = float(input('Enter tPSA: '))
                # qed = float(input('Enter QED: '))

                logp = 2.2
                tpsa = 28.8
                qed = 0.62

            elif if_continue.upper() == 'N':
                exit('Bye bye ~')
            else:
                print('Please enter Y/N !!!!')
                continue

            predictor = Predictor(
                getattr(model, args.decode_type), args.use_cond2dec)

            demo = Demo(args.conditions, predictor, args.decode_type, args.latent_dim,
                        args.max_strlen, args.use_cond2dec, toklen_data, scaler, TRG,
                        (args.logp_lb, args.logp_ub), (args.tpsa_lb, args.tpsa_ub),
                        (args.qed_lb, args.qed_ub), 'beam_search', device)

            demo.inference_from_properties(
                logp, tpsa, qed, num_samples=40, mu=0, std=0.2)

    elif args.test_random:
        # settings
        std_choices = (0.2, 0.4, 0.6, 0.8, 1.0)

        generate_smiles_time = store_properties_time = 0
        total_time = time()

        target_properties = np.array(np.meshgrid(np.linspace(args.logp_lb, args.logp_ub, num=args.num_points),
                                                 np.linspace(
                                                     args.tpsa_lb, args.tpsa_ub, num=args.num_points),
                                                 np.linspace(args.qed_lb, args.qed_ub, num=args.num_points))) \
            .T.reshape(-1, 3)

        print("Predict molecules given all combination of properties")

        bsTool = BeamSearchTool(args.nconds, args.latent_dim,
                                args.max_strlen, model, args.use_cond2dec)
        predictor = ModelPrediction(
            getattr(model, args.decode_type), args.use_cond2dec)

        for std in std_choices:
            storage_path = args.storage_path + f'_std{std}'
            os.makedirs(storage_path, exist_ok=True)

            for i, (logp, tpsa, qed) in enumerate(target_properties):
                properties = (logp, tpsa, qed)

                print(f"\n({i}) Desirable Properties (logP, tPSA, QED): "
                      f"{logp:.2f}, {tpsa:.2f}, {qed:.2f}")

                smiles_path = os.path.join(
                    storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_pre.txt')
                smiles_property_path = os.path.join(
                    storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt')

                if os.path.exists(smiles_property_path):
                    continue  # filter the files already run

                generate_smiles_time -= time()
                generated_smiles = generate_smiles_from_properties(args, bsTool, predictor, properties,
                                                                   TRG, scaler, toklen_data, device,
                                                                   args.num_samplings, mu=0, std=std)
                generate_smiles_time += time()

                with open(smiles_path, 'w') as sample_file:
                    sample_file.write(f"number\tsmiles\n")
                    for i in range(len(generated_smiles)):
                        sample_file.write(f"{i+1}\t{generated_smiles[i]}\n")

                store_properties_time -= time()
                store_properties_from_predicted_smiles(properties,
                                                       property_prediction,
                                                       generated_smiles,
                                                       smiles_property_path,
                                                       args.conditions)
                store_properties_time += time()

                print(f"generateT(s): {generate_smiles_time}\t"
                      f"storepropT(s): {store_properties_time}\t"
                      f"totalT(s): {time() - total_time}")
                os.remove(smiles_path)

            print("Compute metrics for generated smiles of each desirable properties")

            for logp, tpsa, qed in target_properties:
                mean_file_path = os.path.join(
                    storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt')

                with open(mean_file_path, 'w') as metrics_writer:
                    preds = pd.read_csv(os.path.join(
                        storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
                    head, line = generate_metrics_line(
                        preds, train_smiles, args.n_jobs)
                    metrics_writer.write('logp\ttpsa\tqed\t'+head+'\n')
                    metrics_writer.write(
                        f'{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t'+line+'\n')
                    print(head+'\n'+line)

            print("Combine all of the metrics computed before.")

            all_metrics = None

            for i, (logp, tpsa, qed) in enumerate(target_properties):
                preds = pd.read_csv(os.path.join(
                    storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt'), sep='\t')
                all_metrics = pd.concat(
                    [all_metrics, preds], axis=0, ignore_index=True)

            all_metrics = all_metrics.sort_values(by=['logp', 'tpsa', 'qed'])
            all_metrics.to_csv(os.path.join(
                storage_path, 'mean.txt'), sep='\t', index=False)

            print("Compute metrics for smiles of all property combinations")

            # if not os.path.exists(os.path.join(args.storage_path, 'output.txt')):
            all_preds = None
            for logp, tpsa, qed in target_properties:
                preds = pd.read_csv(os.path.join(
                    storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
                all_preds = pd.concat([all_preds, preds], axis=0)
            all_preds = all_preds.reset_index()

            with open(os.path.join(storage_path, 'output.txt'), 'w') as all_metrics_writer:
                head, line = generate_metrics_line(
                    all_preds, train_smiles, args.n_jobs)
                all_metrics_writer.write(head+'\n')
                all_metrics_writer.write(line+'\n')

            print(
                "Compute metrics for smiles of all property combinations except for logP=0.03")

            # if not os.path.exists(os.path.join(args.storage_path, 'output-logp0.03.txt')):
            all_preds = None
            for logp, tpsa, qed in target_properties:
                if logp == 0.03:
                    continue
                preds = pd.read_csv(os.path.join(
                    storage_path, f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
                all_preds = pd.concat([all_preds, preds], axis=0)
            all_preds = all_preds.reset_index()

            with open(os.path.join(storage_path, 'output-logp0.03.txt'), 'w') as all_metrics_writer:
                head, line = generate_metrics_line(
                    all_preds, train_smiles, args.n_jobs)
                all_metrics_writer.write(head+'\n')
                all_metrics_writer.write(line+'\n')

            print("Work Finished")

    else:
        generate_smiles_time = store_properties_time = 0
        os.makedirs(args.storage_path, exist_ok=True)

        total_time = time()
        target_properties = np.array(np.meshgrid(np.linspace(args.logp_lb, args.logp_ub, num=args.num_points),
                                                 np.linspace(
                                                     args.tpsa_lb, args.tpsa_ub, num=args.num_points),
                                                 np.linspace(args.qed_lb, args.qed_ub, num=args.num_points))) \
            .T.reshape(-1, 3)

        bsTool = BeamSearchTool(args.nconds, args.latent_dim,
                                args.max_strlen, model, args.use_cond2dec)
        predictor = ModelPrediction(
            getattr(model, args.decode_type), args.use_cond2dec)

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
            store_properties_from_predicted_smiles(properties,
                                                   property_prediction,
                                                   generated_smiles,
                                                   smiles_property_path,
                                                   args.conditions)
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
                head, line = generate_metrics_line(
                    preds, train_smiles, args.n_jobs)
                metrics_writer.write('logp\ttpsa\tqed\t'+head+'\n')
                metrics_writer.write(
                    f'{logp:.2f}\t{tpsa:.2f}\t{qed:.2f}\t'+line+'\n')

        print("Combine all of the metrics computed before.")

        all_metrics = None

        for i, (logp, tpsa, qed) in enumerate(target_properties):
            preds = pd.read_csv(os.path.join(args.storage_path,
                                             f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt'), sep='\t')
            all_metrics = pd.concat(
                [all_metrics, preds], axis=0, ignore_index=True)

        # all_metrics = pd.concat([properties, all_metrics], axis=1)
        # print(all_metrics.head())
        all_metrics = all_metrics.sort_values(by=['logp', 'tpsa', 'qed'])
        all_metrics.to_csv(os.path.join(args.storage_path,
                           'mean.txt'), sep='\t', index=False)

        print("Compute metrics for smiles of all property combinations")

        # if not os.path.exists(os.path.join(args.storage_path, 'output.txt')):
        all_preds = None
        for logp, tpsa, qed in target_properties:
            preds = pd.read_csv(os.path.join(args.storage_path,
                                f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
            all_preds = pd.concat([all_preds, preds], axis=0)
        all_preds = all_preds.reset_index()

        with open(os.path.join(args.storage_path, 'output.txt'), 'w') as all_metrics_writer:
            head, line = generate_metrics_line(
                all_preds, train_smiles, args.n_jobs)
            all_metrics_writer.write(head+'\n')
            all_metrics_writer.write(line+'\n')

        print(
            "Compute metrics for smiles of all property combinations except for logP=0.03")

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
            head, line = generate_metrics_line(
                all_preds, train_smiles, args.n_jobs)
            all_metrics_writer.write(head+'\n')
            all_metrics_writer.write(line+'\n')

        print("Work Finished")
