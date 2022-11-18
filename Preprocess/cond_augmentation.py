import os
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta


def debug(smiles, props, similar_pairs, input_data, 
          rescaler, conds, tolerances, n_tests=100):
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

        print("check if the props match the original original props...")
        prop1 = rescaler(input_row[[f'src_{c}' for c in conds]], inverse=True)
        prop2 = rescaler(input_row[[f'trg_{c}' for c in conds]], inverse=True)
        assert (prop1.values == prop1_ori.values).any()
        assert (prop2.values == prop2_ori.values).any()


def find_similar_pairs(props, conds, tols):
    props = props.sort_values(by=conds[0])
    similar_pairs = []
    cost_time = -time()
    for id1 in range(len(props)):
        id2 = id1
        while id2 < len(props):
            if abs(props[conds[0]].iloc[id2] - props[conds[0]].iloc[id1]) > tols[0]:
                break
            if abs(props[conds[1]].iloc[id2] - props[conds[1]].iloc[id1]) > tols[1]:
                break
            if abs(props[conds[2]].iloc[id2] - props[conds[2]].iloc[id1]) > tols[2]:
                break
            no1, no2 = props.index[id1], props.index[id2]
            if no1 > no2:
                no1, no2 = no2, no1
            similar_pairs.append((no1, no2))
            id2 += 1
        if id1 % 2000 == 0 or id1 == len(props)-1:
            print("schedule:", id1, end='\r')
    cost_time += time()
    print(f"Total time costed: {str(timedelta(seconds=cost_time))}")
    similar_pairs = pd.DataFrame.from_records(similar_pairs, columns =['no1', 'no2'])
    assert len(similar_pairs) >= len(props)
    return similar_pairs 


# fix bug of merging two dataframes
# https://stackoverflow.com/questions/11976503/how-to-keep-index-when-using-pandas-merge
def prepare_input_data(conds, smiles, props, similar_pairs):
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
        
    input_data = pd.concat([src_smiles_props, trg_smiles_props], axis=1)
    return input_data


def cond_augmentation(args, raw_path, aug_path, rescaler, logger=None):
    """
    augmentation by pairing molecules with similar properties
    """
    tolP1 = (args.logp_ub - args.logp_lb) * args.tolerance
    tolP2 = (args.tpsa_ub - args.tpsa_lb) * args.tolerance
    tolP3 = (args.qed_ub - args.qed_lb) * args.tolerance

    for data_type in ('train', 'validation'):
        LOG = logger(name='augment data by conditions',
                     log_path=os.path.join(aug_path, data_type, "records.log"))

        smiles_path = os.path.join(raw_path, data_type, 'smiles_serial.csv')
        props_path = os.path.join(raw_path, data_type, 'prop_serial.csv')
        pair_path = os.path.join(aug_path, data_type, f'pair_serial_tol{args.tolerance:.2f}.csv')
        input_path = os.path.join(aug_path, f'data_tol{args.tolerance}', f'{data_type}.csv')

        os.makedirs(os.path.join(raw_path, data_type), exist_ok=True)
        os.makedirs(os.path.join(aug_path, data_type), exist_ok=True)
        os.makedirs(os.path.join(aug_path, f'data_tol{args.tolerance}'), exist_ok=True)

        smiles = pd.read_csv(smiles_path, index_col='no')
        props = pd.read_csv(props_path, index_col='no')
        # smiles, props = smiles[:100], props[:100]

        LOG.info("Find similar pairs...")
        LOG.info(f'Tolerance of logP: {tolP1:.3f}, tPSA: {tolP2:.3f}, QED: {tolP3:.3f}')
        similar_pairs = find_similar_pairs(props, args.conditions,
                                           tols=[tolP1, tolP2, tolP3])

        LOG.info(f"# similar pairs: {len(similar_pairs)}")
        LOG.info(f"# augmentation ability: {(len(similar_pairs)/len(smiles)-1)*100:.2f}%")
        similar_pairs.to_csv(pair_path, index=False)

        LOG.info("Prepare input data...")
        rescaled_props = rescaler(props)
        input_data = prepare_input_data(args.conditions, smiles, rescaled_props, similar_pairs)
        
        debug(smiles, props, similar_pairs, input_data,
              rescaler, args.conditions, [tolP1, tolP2, tolP3])
        
        input_data.to_csv(input_path, index=False)

        
