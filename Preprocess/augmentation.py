import os
import pandas as pd
from time import time
from datetime import timedelta

from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file


def create_fp_file(smiFilePath, fpFilePath, 
                   fp_type='Morgan', radius=2, nBits=1024):
    """ create a fingerprint file for all smiles
    the settinggs (fp_type, radius, nBits) are the same
    as those in MOSES when it compute internal diversity
    """
    create_db_file(smiFilePath, fpFilePath,
                   fp_type, {'radius': radius, 'nBits': nBits})


def get_similar_molecular_pairs(smiles_path, pair_path, similarity, n_workers):
    smiles_folder, file_name = smiles_path.rsplit('/', 1)
    smi_file_path = os.path.join(smiles_folder, f'{file_name.split(".")[0]}.smi')
    fp_file_path = os.path.join(smiles_folder, f'{file_name.split(".")[0]}.h5')

    dataset = pd.read_csv(smiles_path)

    print('Create fingerprint file:', fp_file_path)
    if not os.path.exists(smi_file_path):
        dataset.to_csv(smi_file_path, header=None, index=False, sep='\t')
    if not os.path.exists(fp_file_path):
        create_fp_file(smiles_path, fp_file_path)

    fpe = FPSim2Engine(fp_file_path)
    dataset = list(dataset.to_records(index=False))


    buffer = min(2000, len(dataset))
    mol_pairs = []
    start_time = time()
    find_sim_time = build_pairs_time = 0
    first = True

    print(similarity)

    for i, (smi, idx1) in enumerate(dataset):
        find_sim_time -= time()
        results = fpe.similarity(smi, similarity, n_workers=n_workers)        
        find_sim_time += time()
        
        build_pairs_time -= time()
        # results['mol_id'] = pd.to_numeric(results['mol_id'])
        # ids = pd.to_numeric(results['mol_id']).tolist()        
        # mol_pairs.extend([[idx1, idx2] for idx2 in ids])

        mol_pairs.extend([[int(idx1), int(idx2)] for idx2, _ in results])
        
        build_pairs_time += time()

        if (i+1) % buffer == 0 or i == len(dataset)-1:
            end_time = time()
            print(f'Process {buffer} smiles -> {len(mol_pairs)} pairs\t'
                  f'simTime: {str(timedelta(seconds=find_sim_time))}\t'
                  f'buildTime: {str(timedelta(seconds=build_pairs_time))}\t'
                  f'totalTime: {str(timedelta(seconds=end_time-start_time))}\t')
            
            df = pd.DataFrame(mol_pairs, columns=["no1", "no2"])
            if first:
                df.to_csv(pair_path, index=False, mode='a')
                first = False
            else:
                df.to_csv(pair_path, index=False, mode='a', header=False)
            mol_pairs = []


# def data_augmentation(dataset, save_path, similarity, n_jobs):
#     assert 0 <= similarity and similarity <= 1
    
#     n_data = len(dataset)
#     buffer = 100

#     if similarity == 1:
#         df = pd.DataFrame({ 'no1': dataset['no'].tolist(), 'no2': dataset['no'].tolist()})
#         df.to_csv(save_path, index=False)
#         return
    
#     data_smi, data_no = dataset['smiles'].tolist(), dataset['no'].tolist()
#     dataset = list(dataset.to_records(index=False)) # DataFrame to a list of tuples (smiles, no)
#     convert_dict = { i: data_no[i] for i in range(len(data_no)) }

#     start_time = time.time()
#     pairs = []

#     for begin, smiles in enumerate(data_smi):
#         right_smi = data_smi[begin:n_data]

#         with Pool(n_jobs) as p:
#             similarities = list(p.map(partial(tanimoto_similarity, smiles), right_smi))
#             similar_no = [begin + i for i in range(n_data - begin)
#                           if similarities[i] >= similarity]
#             for no in similar_no:
#                 pairs.append([convert_dict[begin], convert_dict[no]])
    
#         if (begin + 1) % buffer == 0 or begin == n_data - 1:
#             print('>>> PROCESSED {:.2f}% - SIMILAR PAIRS: {}\tSIMILARITY: {}\tBUFFER: {}'.format(
#                 (begin + 1)/len(data_smi)*100, len(pairs), similarity, buffer
#             ))
#             df = pd.DataFrame(pairs, columns=["no1", "no2"])
#             if not os.path.exists(save_path):
#                 df.to_csv(save_path, index=False)
#             else:
#                 df.to_csv(save_path, index=False, mode='a', header=False)
#             pairs = []

#     elipsed_time = time.time() - start_time
#     print(">>> ELAPSED TIME:", str(timedelta(seconds=elipsed_time)))
