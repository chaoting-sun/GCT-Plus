import os
import swifter
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from Utils.property import tanimoto_similarity
import matplotlib.pyplot as plt


def validate(smiles, props, similar_pairs, input_data, 
             rescaler, conds, tolerances, similarity, n_tests=100):
    test_id = np.random.choice(len(similar_pairs), n_tests)
    for row_id in test_id:
        # test if the pairs are valid given tolerances
        pair = similar_pairs.iloc[row_id]
        no1, no2 = pair['no1'], pair['no2']
        print(f'test serial: {no1} and {no2}:')
        
        print('check tolerance...')
        prop1_ori, prop2_ori = props.loc[no1], props.loc[no2]
        for i, c in enumerate(conds):
            assert abs(prop1_ori[c] - prop2_ori[c]) <= tolerances[i], \
                f'property: {c}, delta = abs({prop1_ori[c]:.2f} - {prop2_ori[c]:.2f}) > {tolerances[i]:.2f}'

        print("check if the pairs exist in the input_data...")
        input_row = input_data[(input_data.src_no == no1) & (input_data.trg_no == no2)]
        assert len(input_row) == 1, input_row

        print("check if the smiles matches the original original smiles...")
        smi1_ori, smi2_ori = smiles.loc[no1, 'smiles'], smiles.loc[no2, 'smiles']
        smi1, smi2 = input_row['src'].values[0], input_row['trg'].values[0]
        assert smi1 == smi1_ori, f'smi1: {smi1}, smi1_ori: {smi1_ori}'
        assert smi2 == smi2_ori, f'smi1: {smi2}, smi1_ori: {smi2_ori}'

        print("check if the smiles pair has similarity higher than the setting...")
        assert tanimoto_similarity(smi1, smi2) >= similarity

        print("check if the props match the original original props...")
        prop1 = rescaler(input_row[[f'src_{c}' for c in conds]], inverse=True)
        prop2 = rescaler(input_row[[f'trg_{c}' for c in conds]], inverse=True)
        assert (prop1.values == prop1_ori.values).any()
        assert (prop2.values == prop2_ori.values).any()


def plot_similarity(args, aug_path):
    print('plot similarity...')
    data_path = os.path.join(aug_path, f'data_tol{args.tolerance}')
    df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    df = df.sample(10000)
    df = df.loc[df.src_no != df.trg_no]
    
    # df['similarity'] = calc_similarity(df['src'].to_numpy(),
    #                                    df['trg'].to_numpy())
    # df = df[:10000]
    df['similarity'] = df.apply(lambda r: tanimoto_similarity(r['src'], r['trg']), axis=1)
    print(df)

    df_low = df.loc[df.similarity < 0.5]
    print(df_low['src'].iloc[10], df_low['trg'].iloc[10])

    plt.figure(figsize=(7,5))
    plt.xlim((0,1))
    plt.title("Similarity of Train Pairs (tol=0.01)", fontsize=20)
    plt.xlabel("Similarity", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    df['similarity'].plot.kde(bw_method=0.2)
    plt.tight_layout()
    plt.savefig('./sim.png')
    
    exit() 


# def find_similar_prop_pairs(props, conds, tols):
#     props = props.sort_values(by=conds[0])
#     similar_pairs = []
#     cost_time = -time()
#     for id1 in range(len(props)):
#         id2 = id1
#         while id2 < len(props):
#             if abs(props[conds[0]].iloc[id2] - props[conds[0]].iloc[id1]) > tols[0]:
#                 break
#             if abs(props[conds[1]].iloc[id2] - props[conds[1]].iloc[id1]) > tols[1]:
#                 break
#             if abs(props[conds[2]].iloc[id2] - props[conds[2]].iloc[id1]) > tols[2]:
#                 break
#             no1, no2 = props.index[id1], props.index[id2]
#             similar_pairs.append((no1, no2))
#             if no1 != no2:
#                 similar_pairs.append((no2, no1))
#             id2 += 1
#         if id1 % 2000 == 0 or id1 == len(props)-1:
#             print("schedule:", id1, end='\r')
#     cost_time += time()
#     print(f"Total time costed: {str(timedelta(seconds=cost_time))}")
#     similar_pairs = pd.DataFrame.from_records(similar_pairs, columns =['no1', 'no2'])
#     assert len(similar_pairs) >= len(props)
#     return similar_pairs 


def _find_similar_pairs(props_smiles, conds, sim_tol, prop_tols, LOG):
    props_smiles = props_smiles.sort_values(by=conds[0])
    similar_pairs = []
    for id1 in range(len(props_smiles)):
        id2 = id1
        while id2 < len(props_smiles):
            if abs(props_smiles[conds[0]].iloc[id2] - props_smiles[conds[0]].iloc[id1]) > prop_tols[0]:
                break
            if abs(props_smiles[conds[1]].iloc[id2] - props_smiles[conds[1]].iloc[id1]) > prop_tols[1]:
                break
            if abs(props_smiles[conds[2]].iloc[id2] - props_smiles[conds[2]].iloc[id1]) > prop_tols[2]:
                break
            if tanimoto_similarity(props_smiles['smiles'].iloc[id1], props_smiles['smiles'].iloc[id2]) < sim_tol:
                break
            
            no1, no2 = props_smiles.index[id1], props_smiles.index[id2]
            similar_pairs.append((no1, no2))
            if no1 != no2:
                similar_pairs.append((no2, no1))
            id2 += 1
        if id1 % 5000 == 0 or id1 == len(props_smiles)-1:
            LOG.info(f"schedule: {id1}, "
                     f"# pairs: {len(similar_pairs)}, "
                     f"aug-ratio: {len(similar_pairs)/(id1+1)-1:.2f} %")
    return pd.DataFrame.from_records(similar_pairs, columns =['no1', 'no2'])


# fix bug of merging two dataframes
# https://stackoverflow.com/questions/11976503/how-to-keep-index-when-using-pandas-merge
def _prepare_input_data(smiles, props, similar_pairs, conds):
    def prepare_smiles_props(smiles, props, smi_type, smi_no):
        props = props.rename(columns={ c: f'{smi_type}_{c}' for c in conds })        
        smiles = smiles.reindex(similar_pairs[smi_no]) # select rows by index values
        smiles_props = smiles.reset_index().merge(
            props, how="left", left_on=smi_no, right_on='no').set_index(smi_no)
        smiles_props = smiles_props.reset_index()
        smiles_props = smiles_props.rename(columns={ 'smiles': smi_type,
                                                     smi_no: f'{smi_type}_no' })
        return smiles_props

    src_smiles_props = prepare_smiles_props(smiles, props, 'src', 'no1')
    trg_smiles_props = prepare_smiles_props(smiles, props, 'trg', 'no2')
    return pd.concat([src_smiles_props, trg_smiles_props], axis=1)


def _build_paths(raw_path, aug_path, data_type, tolerance, similarity):
    data_raw_path = os.path.join(raw_path, data_type)
    data_aug_path = os.path.join(aug_path, data_type)

    train_path = os.path.join(aug_path, f'data_sim{similarity:.2f}_tol{tolerance:.2f}')
    
    os.makedirs(data_raw_path, exist_ok=True)
    os.makedirs(data_aug_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)

    train_data_path = os.path.join(train_path, f'{data_type}.csv')
    pair_serial_path = os.path.join(data_aug_path, f'pair_serial_sim{similarity:.2f}_tol{tolerance:.2f}.csv')

    return data_raw_path, data_aug_path, train_data_path, pair_serial_path


def _get_props_tolerance(args):
    tolP1 = (args.logp_ub - args.logp_lb) * args.tolerance
    tolP2 = (args.tpsa_ub - args.tpsa_lb) * args.tolerance
    tolP3 = (args.qed_ub - args.qed_lb) * args.tolerance
    return [tolP1, tolP2, tolP3]


def _get_raw_data(raw_data_path):
    smiles = pd.read_csv(os.path.join(raw_data_path, 'smiles_serial.csv'), index_col='no')
    props = pd.read_csv(os.path.join(raw_data_path, 'prop_serial.csv'), index_col='no')
    return smiles, props, pd.concat([smiles, props], axis=1)


def data_augmentation(args, data_type, raw_path, aug_path, rescaler, logger):
    data_raw_path, data_aug_path, train_data_path, pair_serial_path = _build_paths(raw_path,
                                                                                   aug_path,
                                                                                   data_type,
                                                                                   args.tolerance,
                                                                                   args.similarity)
    LOG = logger(name='data augmentation', log_path=os.path.join(data_aug_path, "records.log"))
    
    props_tol = _get_props_tolerance(args)
    smiles, props, props_smiles = _get_raw_data(data_raw_path)

    LOG.info(f"Find similar pairs:\t"
             f"tol = {props_tol[0]:.2f}, {props_tol[1]:.2f}, {props_tol[2]:.2f}\t"
             f"sim = {args.similarity}")
    similar_pairs = _find_similar_pairs(props_smiles,
                                        args.conditions,
                                        args.similarity,
                                        props_tol,
                                        LOG)
    LOG.info(f"# similar pairs: {len(similar_pairs)}")

    LOG.info(f"# augmentation ability: {(len(similar_pairs)/len(props_smiles)-1)*100:.2f}%")
    similar_pairs.to_csv(pair_serial_path, index=False)

    LOG.info("Prepare train data...")
    rescaled_props = rescaler(props)
    train_data = _prepare_input_data(smiles,
                                     rescaled_props,
                                     similar_pairs,
                                     args.conditions)

    LOG.info("Sample pairs to validate...")
    validate(smiles, props, similar_pairs, train_data, rescaler,
             args.conditions, props_tol, args.similarity)
    
    train_data.to_csv(train_data_path, index=False)



