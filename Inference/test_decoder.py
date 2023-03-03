import os
import numpy as np
import pandas as pd
from Inference.utils import prepare_generator
import torch
from Utils.properties import tanimoto_similarity as similarity_fcn


property_peaks = [3.075,93.411,0.609]


def distance_fcn(z1, z2):
    return torch.sqrt(torch.sum((z2 - z1)**2)).item()
 
 
def test_decoder(args, toklen_data, scaler, SRC, TRG, device):
    for epoch in args.epoch_list:
        args.use_model_path = os.path.join(args.train_path,
                                        args.model_name,
                                        f'model_{epoch}.pt')

        generator = prepare_generator(args, SRC, TRG,
                                      toklen_data, scaler, device)

        max_length = 80
        similarity_list = []
        distance_list = []
        cnt = 0
        n_samples = 1000

        props_t = np.array([property_peaks])
        props_t = np.tile(props_t, (2, 1))

        while cnt < n_samples:
            zs, _ = generator.sample_z_from_data(n=2)
            
            smiles, *_ = generator.sample_smiles(props_t, zs)
            sim = similarity_fcn(smiles[0], smiles[1])
            if sim == None:
                continue
            print(cnt, smiles)

            pad = torch.zeros((1,abs(max_length-zs[0].size(1)), zs[0].size(2)), dtype=torch.long)
            zs[0] = torch.concat([zs[0], pad], axis=1)
            pad = torch.zeros((1,abs(max_length-zs[1].size(1)), zs[1].size(2)), dtype=torch.long)
            zs[1] = torch.concat([zs[1], pad], axis=1)

            similarity_list.append(sim)
            distance_list.append(distance_fcn(zs[0], zs[1]) / max_length)
            cnt += 1
        
    df = pd.DataFrame({ 'similarity': similarity_list, 'distance': distance_list })
    df['distance'] = df['distance'] / df['distance'].max()
    print(df)
    df.to_csv('./4.csv')