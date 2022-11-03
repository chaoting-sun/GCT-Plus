import os
import numpy as np
import pandas as pd
from time import time
from rdkit import Chem

from Utils.property import property_prediction
from Inference.metrics import get_all_metrics


N_SAMPLINGS = 5
N_EACH_PROP = 2


def sample_z_uniformly(minmax1, minmax2, minmax3):
    target_props = np.array(np.meshgrid(np.linspace(minmax1[0], minmax1[1], num=N_EACH_PROP),
                                        np.linspace(
                                            minmax2[0], minmax2[1], num=N_EACH_PROP),
                                        np.linspace(minmax3[0], minmax3[1], num=N_EACH_PROP))) \
        .T.reshape(-1, 3)
    return target_props


def store_properties(conditions, gen_smiles, smiles_path,
                     logp_t, tpsa_t, qed_t, logger):
    with open(smiles_path, "w", buffering=10) as ptr:
        ptr.write(f"number\tsmiles\tvalid\t"
                  f"logp_t\ttpsa_t\tqed_t\t"
                  f"logp_p\ttpsa_p\tqed_p\n")

        for i in range(len(gen_smiles)):
            mol = Chem.MolFromSmiles(gen_smiles[i])
            if mol is not None:
                valid = 1
                logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
                                         for c in conditions)
            else:
                valid = 0
                logp_p = tpsa_p = qed_p = np.nan

            line = f"{i+1}\t{gen_smiles[i]}\t{valid}\t"     \
                   f"{logp_t:.2f}\t{tpsa_t:.2f}\t{qed_t:.2f}\t" \
                   f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}"

            ptr.write(line+"\n")
            logger.info(
                f'- {gen_smiles[i]:<50} -> {logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}')


def store_metrics(logp_t, tpsa_t, qed_t, smiles_path,
                  metrics_path, train_smiles, n_jobs, logger):
    with open(metrics_path, 'w') as ptr:
        gen_smiles = pd.read_csv(smiles_path, sep='\t')
        all_metrics = get_all_metrics(gen_smiles, train_smiles, n_jobs)
        header, body = print_all_metrics(all_metrics)
        ptr.write('logp\ttpsa\tqed\t'+header+'\n')
        ptr.write(f'{logp_t:.2f}\t{tpsa_t:.2f}\t{qed_t:.2f}\t'+body+'\n')


def print_all_metrics(metrics):
    header = 'valid\tunique\tnovel\tintDiv\t' \
             'logpMAE\ttpsaMAE\tqedMAE\t'     \
             'logpMSE\ttpsaMSE\tqedMSE\t'     \
             'logpMAX\ttpsaMAX\tqedMAX\t'     \
             'logpMIN\ttpsaMIN\tqedMIN\t'     \
             'logpAARD\ttpsaAARD\tqedAARD\t'  \
             'logpAMSD\ttpsaAMSD\tqedAMSD'

    body = f"{metrics['valid']:.4f}\t"   \
           f"{metrics['unique']:.4f}\t"  \
           f"{metrics['novel']:.4f}\t"   \
           f"{metrics['intDiv']:.4f}\t"  \
        \
           f"{metrics['logpErr']['mae']:.4f}\t"  \
           f"{metrics['tpsaErr']['mae']:.4f}\t"  \
           f"{metrics['qedErr']['mae']:.4f}\t"   \
        \
           f"{metrics['logpErr']['mse']:.4f}\t"  \
           f"{metrics['tpsaErr']['mse']:.4f}\t"  \
           f"{metrics['qedErr']['mse']:.4f}\t"   \
        \
           f"{metrics['logpErr']['max']:.4f}\t"  \
           f"{metrics['tpsaErr']['max']:.4f}\t"  \
           f"{metrics['qedErr']['max']:.4f}\t"   \
        \
           f"{metrics['logpErr']['min']:.4f}\t"  \
           f"{metrics['tpsaErr']['min']:.4f}\t"  \
           f"{metrics['qedErr']['min']:.4f}"   \

    return header, body


def generate_uniformly(args, smiles_generator, train_smiles, logger):
    LOG = logger('generate_uniformly',
                 log_path=os.path.join(args.storage_path, "generate_uniformly.log"))

    target_props = sample_z_uniformly((args.logp_lb, args.logp_ub),
                                      (args.tpsa_lb, args.tpsa_ub),
                                      (args.qed_lb, args.qed_ub))

    sample_T = property_T = metrics_T = 0
    total_T = time()

    LOG.info(f"Generate SMILES, compute properties/metrics")

    for i, props in enumerate(target_props):
        logp_t, tpsa_t, qed_t = props

        LOG.info(
            f"{i:<5} logP,tPSA,QED: {logp_t:.2f}, {tpsa_t:.2f}, {qed_t:.2f}")
        smiles_path = os.path.join(args.storage_path,
                                   f'{logp_t:.2f}_{tpsa_t:.2f}_{qed_t:.2f}.txt')
        metrics_path = os.path.join(args.storage_path,
                                    f'{logp_t:.2f}_{tpsa_t:.2f}_{qed_t:.2f}_mean.txt')

        LOG.info("- Sample SMILES...")

        sample_T -= time()
        sampled_smiles = []
        for _ in range(N_SAMPLINGS):
            smiles, toklen_gen, toklen = smiles_generator.sample_smiles([
                                                                        props])
            sampled_smiles.append(smiles)
            LOG.info(smiles)
        sample_T += time()

        LOG.info("- Store SMILES properties...")

        property_T -= time()
        store_properties(args.conditions, sampled_smiles, smiles_path,
                         logp_t, tpsa_t, qed_t, LOG)
        property_T += time()

        LOG.info("- Store metrics...")

        metrics_T -= time()
        store_metrics(logp_t, tpsa_t, qed_t, smiles_path, metrics_path,
                      train_smiles, args.n_jobs, LOG)
        metrics_T += time()

        LOG.info(f"sampleT(s): {sample_T:.1f}\t"
                 f"propertyT(s): {property_T:.1f}\t"
                 f"metricsT(s): {metrics_T:.1f}\t"
                 f"totalT(s): {(time() - total_T):.1f}")

    LOG.info("[ Combine all computed metrics ]")

    all_metrics = None
    for i, (logp, tpsa, qed) in enumerate(target_props):
        preds = pd.read_csv(os.path.join(args.storage_path,
                                         f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt'), sep='\t')
        all_metrics = pd.concat([all_metrics, preds],
                                axis=0, ignore_index=True)
    all_metrics = all_metrics.sort_values(by=['logp', 'tpsa', 'qed'])
    all_metrics.to_csv(os.path.join(args.storage_path,
                       'mean.txt'), sep='\t', index=False)

    LOG.info("[ Compute metrics for all sampled smiles ]")

    all_gens = None
    for i, (logp, tpsa, qed) in enumerate(target_props):
        gens = pd.read_csv(os.path.join(args.storage_path,
                                        f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
        all_gens = pd.concat([all_gens, gens], axis=0)
    all_gens = all_gens.reset_index()

    with open(os.path.join(args.storage_path, 'output.txt'), 'w') as ptr:
        all_metrics = get_all_metrics(all_gens, train_smiles, args.n_jobs)
        header, body = print_all_metrics(all_metrics)
        ptr.write(header+'\n')
        ptr.write(body+'\n')

    LOG.info("Finished.")
