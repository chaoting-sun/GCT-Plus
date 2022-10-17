import os
import numpy as np
import pandas as pd
from time import time
from rdkit import Chem

from Utils.property import property_prediction
from Inference.metrics import all_metrics as compute_metrics


N_SAMPLINGS = 5
N_EACH_PROP = 2


def sample_z_uniformly(minmax1, minmax2, minmax3):
    target_props = np.array(np.meshgrid(np.linspace(minmax1[0], minmax1[1], num=N_EACH_PROP),
                                        np.linspace(
                                            minmax2[0], minmax2[1], num=N_EACH_PROP),
                                        np.linspace(minmax3[0], minmax3[1], num=N_EACH_PROP))) \
        .T.reshape(-1, 3)
    return target_props


def store_properties(conditions, sampled_smiles, smiles_path,
                     logp_t, tpsa_t, qed_t, logger):
    with open(smiles_path, "w", buffering=10) as ptr:
        ptr.write(f"number\tsmiles\tvalid\t"
                  f"logp_t\ttpsa_t\tqed_t\t"
                  f"logp_p\ttpsa_p\tqed_p\n")

        for i in range(len(sampled_smiles)):
            mol = Chem.MolFromSmiles(sampled_smiles[i])
            if mol is not None:
                valid = 1
                logp_p, tpsa_p, qed_p = (property_prediction[c](mol)
                                         for c in conditions)
            else:
                valid = 0
                logp_p = tpsa_p = qed_p = np.nan

            line = f"{i+1}\t{sampled_smiles[i]}\t{valid}\t"     \
                   f"{logp_t:.2f}\t{tpsa_t:.2f}\t{qed_t:.2f}\t" \
                   f"{logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}"

            ptr.write(line+"\n")
            logger.info(f'- {sampled_smiles[i]:<50} -> {logp_p:.2f}\t{tpsa_p:.2f}\t{qed_p:.2f}')


def store_metrics(logp_t, tpsa_t, qed_t, smiles_path,
                  metrics_path, train_smiles, n_jobs, logger):
    with open(metrics_path, 'w') as ptr:
        sampled_smiles = pd.read_csv(smiles_path, sep='\t')
        head, line = compute_metrics(sampled_smiles, train_smiles, n_jobs)
        ptr.write('logp\ttpsa\tqed\t'+head+'\n')
        
        metrics_line = f'{logp_t:.2f}\t{tpsa_t:.2f}\t{qed_t:.2f}\t'+line+'\n'
        ptr.write(metrics_line)
        logger.info(metrics_line)


def generate_uniformly(args, logger, smiles_generator, train_smiles):
    target_props = sample_z_uniformly((args.logp_lb, args.logp_ub),
                                      (args.tpsa_lb, args.tpsa_ub),
                                      (args.qed_lb, args.qed_ub))

    sample_T = property_T = metrics_T = 0
    total_T = time()

    logger.info(f"Generate SMILES, compute properties/metrics")

    for i, props in enumerate(target_props):
        logp_t, tpsa_t, qed_t = props

        logger.info(f"{i:<5} logP,tPSA,QED: {logp_t:.2f}, {tpsa_t:.2f}, {qed_t:.2f}")
        smiles_path = os.path.join(args.storage_path,
                                   f'{logp_t:.2f}_{tpsa_t:.2f}_{qed_t:.2f}.txt')
        metrics_path = os.path.join(args.storage_path,
                                    f'{logp_t:.2f}_{tpsa_t:.2f}_{qed_t:.2f}_mean.txt')

        logger.info("- Sample SMILES...")
        
        sample_T -= time()
        sampled_smiles = []
        for _ in range(N_SAMPLINGS):
            smiles, toklen_gen, toklen = smiles_generator.sample_smiles([props])
            sampled_smiles.append(smiles)
            logger.info(smiles)
        sample_T += time()

        logger.info("- Store SMILES properties...")

        property_T -= time()
        store_properties(args.conditions, sampled_smiles, smiles_path,
                         logp_t, tpsa_t, qed_t, logger)
        property_T += time()

        logger.info("- Store metrics...")

        metrics_T -= time()
        store_metrics(logp_t, tpsa_t, qed_t, smiles_path, metrics_path,
                      train_smiles, args.n_jobs, logger)
        metrics_T += time()

        logger.info(f"sampleT(s): {sample_T:.1f}\t"
                    f"propertyT(s): {property_T:.1f}\t"
                    f"metricsT(s): {metrics_T:.1f}\t"
                    f"totalT(s): {(time() - total_T):.1f}")

    logger.info("[ Combine all computed metrics ]")

    all_metrics = None
    for i, (logp, tpsa, qed) in enumerate(target_props):
        preds = pd.read_csv(os.path.join(args.storage_path,
                                         f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}_mean.txt'), sep='\t')
        all_metrics = pd.concat([all_metrics, preds],
                                axis=0, ignore_index=True)
    all_metrics = all_metrics.sort_values(by=['logp', 'tpsa', 'qed'])
    all_metrics.to_csv(os.path.join(args.storage_path,
                       'mean.txt'), sep='\t', index=False)

    logger.info("[ Compute metrics for all sampled smiles ]")

    all_sampled_smiles = None
    for i, (logp, tpsa, qed) in enumerate(target_props):
        sampled_smiles = pd.read_csv(os.path.join(args.storage_path,
                                                  f'{logp:.2f}_{tpsa:.2f}_{qed:.2f}.txt'), sep='\t')
        all_sampled_smiles = pd.concat(
            [all_sampled_smiles, sampled_smiles], axis=0)
    all_sampled_smiles = all_sampled_smiles.reset_index()

    with open(os.path.join(args.storage_path, 'output.txt'), 'w') as ptr:
        head, line = compute_metrics(all_sampled_smiles, train_smiles, args.n_jobs)
        ptr.write(head+'\n')
        ptr.write(line+'\n')

    logger.info("Finished.")
