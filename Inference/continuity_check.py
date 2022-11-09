import os
import numpy as np
import pandas as pd
import torch
from rdkit.Chem import MolFromSmiles, MolToSmiles
from moses.metrics import SNNMetric
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.dataset import get_dataloader
from Utils.property import get_mol, get_smiles, property_prediction
from Inference.metrics import get_all_metrics, get_snn_from_mol, get_interval_diversity, get_basic_metrics

# pd.to_csv() -> pd.read_csv(.., index_col=[0])


def rand_z(n, toklen, latent_dim):
    return torch.Tensor(np.random.normal(
           size=(n, toklen, latent_dim)))


def distance(z1, z2):
    return torch.sqrt(torch.sum((z2-z1)**2)).item()


def snnOf2MolGroups(molList1, molList2):
    return SNNMetric()(gen=molList1, ref=molList2)
  

def digits2smiles(vocab, digit_type='src'):
    def translate(digits):
        smi_list = [vocab.itos[d] for d in digits if d != vocab.stoi["<pad>"]]
        if digit_type == 'src':
            return ''.join(smi_list)
        elif digit_type == 'trg':
            return ''.join(smi_list[1:-1])
        else:
            exit(f'No digit_type: {digit_type}')
    return translate


def snn_of_mol_groups(mol_groups):
    n_groups = len(mol_groups)
    snn_start = np.ones((n_groups,), dtype=np.float32)
    snn_prev = np.full((n_groups,), np.nan, dtype=np.float32)
    
    for i in range(1, n_groups):
        snn_start[i] = get_snn_from_mol(mol_groups[0], mol_groups[i])
        snn_prev[i] = get_snn_from_mol(mol_groups[i-1], mol_groups[i])
    return snn_start, snn_prev


def get_z_from_data(args, LOG, smiles_generator, fields, SRC, device, data_type="train", n=2):
    infile_path = os.path.join(
        args.data_path, 'aug', 'data_sim1.00', f"{data_type}_tiny.csv")
    oufile_path = os.path.join(
        args.data_path, 'aug', 'data_sim1.00', f"{data_type}_{n}.csv")

    LOG.info(f"Create new data path: {oufile_path}")
    df = pd.read_csv(infile_path)
    df = df.sample(n=n)
    df.to_csv(oufile_path, index=False)

    LOG.info(f"Get dataloader, #SMILES: {n}")
    dataloader, nbatches = get_dataloader(args.conditions, oufile_path,
                                          fields, SRC.vocab.stoi['<pad>'],
                                          args.max_strlen, device, batch_size=1)

    SRCs = []
    Zs = torch.empty((n, args.max_strlen, args.latent_dim),
                     dtype=torch.float32).to(device)
    Cs = torch.empty((n, 3), dtype=torch.float32).to(device)

    LOG.info(f"Generate Z from {n} source SMILES.")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch.src.requires_grad, batch.econds.requires_grad = False, False
            z = smiles_generator.get_z_from_src(batch.src, batch.econds)[0]

            SRCs.append(batch.src[0])
            Zs[i], Cs[i] = z, batch.econds[0]

    LOG.info(f"Remove data path: {oufile_path}")
    # os.remove(oufile_path)
    return SRCs, Zs, Cs


def get_z_points(toklen, latent_dim, folder, filename=("z1.pt", "z2.pt")):
    zs = []
    for name in filename:
        zpath = os.path.join(folder, name)
        if not os.path.exists(zpath):
            z = rand_z(1, toklen, latent_dim)
            torch.save(z, zpath)
        zs.append(torch.load(os.path.join(folder, name)))
    return zs


def sample_smiles(z, props, n_samples, generator, std=0.4):
    assert z.dim() == 3
    zall = z.repeat(n_samples, 1, 1)
    zall += torch.empty_like(zall).normal_(mean=0, std=std)
    props = np.reshape(np.array(props), (1,3))
    props = np.repeat(props, n_samples, axis=0)
    smiles = generator.sample_smiles(props, zall, transform=True)[0]
    return smiles


def property_from_smiles(smiles, props_predictor, n_jobs=1):
    """
    Compute validity and properties of a
    list of smiles and return a pd.DataFrame
    """
    with Pool(n_jobs) as pool:
        props = pool.map(props_predictor, smiles)
    valids, props = map(lambda x: np.asarray(x), zip(*props))
    valids = pd.DataFrame(data={'valid': valids})
    props = pd.DataFrame(props, columns=["logp_p","tpsa_p","qed_p"])
    props = pd.concat([valids, props], axis=1)
    
    smi = pd.DataFrame(data={'smiles': smiles})
    smi = pd.concat([smi, props], axis=1)
    return props


def props_predictor_wrapper(predictor, conditions):
    def props_predictor(smiles):
        mol = MolFromSmiles(smiles)
        if mol is not None:
            valid = 1
            props = [predictor[c](mol) for c in conditions]
        else:
            valid = 0
            props = [np.nan]*len(conditions)
        return valid, props
    return props_predictor


def percent_not_intersected(smis_groups):
    not_intersected = []
    for i in range(len(smis_groups)):
        if i == 0:
            adjacent_smis = set(smis_groups[i+1])
        elif i == len(smis_groups)-1:
            adjacent_smis = set(smis_groups[i-1])
        else:
            adjacent_smis = set(smis_groups[i-1]) | set(smis_groups[i+1])
        intersected_part = set(smis_groups[i]).intersection(adjacent_smis)
        not_intersected.append(1-len(intersected_part)/len(adjacent_smis))
    return not_intersected


def compute_metrics(smiles, train_smiles, n_jobs):
    with Pool(n_jobs) as pool:
        mols = pool.map(get_mol, smiles)
        mols = [mol for mol in mols if mol]
    
    basic_metrics = get_basic_metrics(smiles, train_smiles, n_jobs)
    
    with Pool(n_jobs) as pool:
        smis = pool.map(get_smiles, mols) # canonical smiles
    
    metrics = {
        "valid": basic_metrics["valid"],
        "unique": basic_metrics["unique"],
        "novel": basic_metrics["novel"],
        "intDiv": basic_metrics["intDiv"]
    }
    return metrics, smis, mols


def store_metrics_from_smiles_file(save_folder, n_steps,
                                   train_smiles, n_jobs):
    z1 = torch.load(os.path.join(save_folder, "z1.pt"))    
    z2 = torch.load(os.path.join(save_folder, "z2.pt"))    
    zs_dist = distance(z1, z2)

    metrics = []
    smis_groups = []
    mols_groups = []
    for i in range(n_steps+1):
        smiles = pd.read_csv(os.path.join(save_folder, 
                 f'z1z2_{i}.csv'))
        smiles = smiles["smiles"]
        m, smis, mols = compute_metrics(smiles, train_smiles, n_jobs)
        metrics.append(m)
        smis_groups.append(smis)
        mols_groups.append(mols)
        
    snn_start, snn_prev = snn_of_mol_groups(mols_groups)
    not_intersected = percent_not_intersected(smis_groups)

    df = pd.DataFrame({
        "distance_start": [zs_dist/n_steps*i for i in range(n_steps+1)],
        "distance_previous": [np.nan] + [zs_dist/n_steps]*n_steps,
        "validity": [m['valid'] for m in metrics],
        "uniqueness": [m['unique'] for m in metrics],
        "novelty": [m['novel'] for m in metrics],
        "internal_diversity": [m['intDiv'] for m in metrics],
        "snn_start": snn_start,
        "snn_previous": snn_prev,
        "percent_not_intersect": not_intersected
    })
    df.to_csv(os.path.join(save_folder, "statistics.csv"))


def continuity_check(generator, latent_dim, conditions, storage_path, properties,
                     toklen, n_steps, n_samples, n_jobs, train_smiles, logger):
    """Continuity check for the latent space:
    Check if the latent space is continuous by sampling
    from several consecutive equal-spacing points between
    two latent space.

    validities: the validity of each smiles group   
    intersections: the percentage of smiles not in the adjacent smiles group(s)
    snn_start: snn between the first smiles group and the other ones. 
    snn_prev: snn between two consecutive smiles groups
    """ 
    
    save_folder = os.path.join(storage_path, f"toklen{toklen}")
    os.makedirs(save_folder, exist_ok=True)

    # store_metrics_of_transvae_gen_smiles(save_folder, n_steps, n_jobs)

    LOG = logger('continuity_check', log_path=os.path.join(save_folder, "records.log"))

    zs = get_z_points(toklen, latent_dim, save_folder)
    zs_vec = zs[1] - zs[0]
    zs_dist = distance(zs[1], zs[0])
    LOG.info(f"Distance between 2 zs: {zs_dist}")

    props_predictor = props_predictor_wrapper(property_prediction, conditions)
    
    props_t = np.reshape(np.array(properties), (1,3))
    props_t = np.repeat(props_t, n_samples, axis=0)
    props_t = pd.DataFrame(props_t, columns=["logp_t","tpsa_t","qed_t"])

    std = (zs_dist/n_steps * 0.5) * 0.5
    std = std if std < 1 else 1
    LOG.info(f"std: {std}")

    assert (zs[0] + zs_vec == zs[1]).any()
    
    for i in range(n_steps+1):
        LOG.info(f"sample smiles {i} / {n_steps}")
        if os.path.exists(os.path.join(save_folder, f"z1z2_{i}.csv")):
            continue
        
        z = zs[0] + (zs_vec / n_steps) * i
        smiles = sample_smiles(z, properties, n_samples, generator, std=std)
        smiles = pd.DataFrame(smiles, columns=['smiles'])
        props_p = property_from_smiles(smiles["smiles"], props_predictor, n_jobs)
        # print(smiles)
        # print(props_p)
        # exit()
        smiles_props = pd.concat([smiles, props_t, props_p], axis=1)
        smiles_props.to_csv(os.path.join(save_folder, f"z1z2_{i}.csv"))
        
        LOG.info("check 10 SMILES: " + ",".join(smiles['smiles'].iloc[:10]))
    
    LOG.info("store metrics from smiles file...")
    store_metrics_from_smiles_file(save_folder, n_steps,
                                   train_smiles, n_jobs)
    LOG.info("execution finished.")


def store_metrics_of_transvae_gen_smiles(save_folder, n_steps, n_jobs):
    """Validation on Transvae:

    sample 50 equal-spacing latent spaces between 2 random sampled zs.
    decoder samples 100 smiles by greedy algorithm. Each z adds a gaussian
    with an average 0 and std equal to 0.7 of half of the distance of two
    consecutive zs. (0.7*(distOf2Zs/2))
    """
    data_path = "/fileserver-gamma/chaoting/ML/dataset/zinc_data/zinc_train.txt"
    save_folder = "/fileserver-gamma/chaoting/ML/TransVAE/continuity-check/"
    
    train_smiles = pd.read_csv(data_path)
    train_smiles = train_smiles["smile"].tolist()
    
    store_metrics_from_smiles_file(save_folder, n_steps,
                                   train_smiles, n_jobs)
    exit(0)
    