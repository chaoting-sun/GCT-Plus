import os
import numpy as np
import pandas as pd

from Utils.dataset import get_dataloader


def get_z_from_data(args, LOG, smiles_generator, fields, TRG, device, data_type="train", n=2):
    infile_path = os.path.join(data_path, 'aug', 'data_sim1.00', f"{data_type}_tiny.csv")
    oufile_path = os.path.join(data_path, 'aug', 'data_sim1.00', f"{data_type}_{n}.csv")

    df = pd.read_csv(infile_path)
    df = df.sample(n=n)
    df.to_csv(oufile_path)

    dataloader, nbatches = get_dataloader(args.conditions, oufile_path,
                                          fields, TRG.vocab.stoi['<pad>'],
                                          args.max_strlen, device, batch_size=1)
    
    Zs = torch.empty((n, args.max_strlen, args.latent_dim), dtype=torch.float32)
    for i, batch in enumerate(dataloader):
        z, mu, log_var = smiles_generator.get_z_from_src(src, econds)
        print(z)
        Zs[i] = z
    os.remove(oufile_path)
    return Zs

def varying_z_generate(args, smiles_generator, fields, device, logger, TRG):
    LOG = logger(name="varying_z_generate", log_path="./test.log")
    np.random.seed(0)
    Zs = get_z_from_data(args, LOG, smiles_generator, fields, TRG, device)
    print((Zs[0] - Zs[0]))



