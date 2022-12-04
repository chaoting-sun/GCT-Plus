import os
import numpy as np
import pandas as pd
import torch
from torchtext import data
from rdkit.Chem import MolFromSmiles
from moses.metrics import SNNMetric
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.dataset import get_dataloader
from Utils.property import get_mol, get_smiles, property_prediction
from Inference.metrics import get_all_metrics, get_snn_from_mol, get_basic_metrics, print_all_metrics
from Utils.property import props_predictor_wrapper, predict_props
from Utils.dataset import to_dataloader
from Inference.utils import augment_props, augment_z


# def augment_props(n_samples, props, varId=None,
#                   bound=None, std=None, toTensor=False):
#     props = np.array(props).reshape((1,3))
#     props = np.repeat(props, n_samples, axis=0)
#     if std:
#         props[:, varId] += np.random.normal(0, std, (n_samples,))
#         for i in range(n_samples):
#             props[i, varId] = min(props[i, varId], bound[1])
#             props[i, varId] = max(props[i, varId], bound[0])     
#     return props


# def augment_z(n_samples, z, std=None):
#     assert z.dim() == 3
#     z = z.repeat(n_samples, 1, 1)
#     if std:
#         z += torch.empty_like(z).normal_(mean=0, std=std)
#         return z
#     return z


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



def sample_smiles(z, props, n_samples, generator, zstd=None, pstd=None):
    assert z.dim() == 3
    
    zall = z.repeat(n_samples, 1, 1)
    if zstd:
        zall += torch.empty_like(zall).normal_(mean=0, std=zstd)
    props = np.array(props).reshape((1,3))
    props = np.repeat(props, n_samples, axis=0)
    if pstd:
        props += np.random.normal(0, pstd, (n_samples, 3))
    props = torch.Tensor(props)
    smiles = generator.sample_smiles(props, zall, transform=True)[0]
    return smiles, zall, props


def property_from_smiles(smiles, n_jobs=1):
    """
    Compute validity and properties of a
    list of smiles and return a pd.DataFrame
    """
    with Pool(n_jobs) as pool:
        props = pool.map(predict_props, smiles)
    valids = [0 if np.nan in p else 1 for p in props]
    
    valids = pd.DataFrame(data={'valid': valids})
    props = pd.DataFrame(props, columns=["logp_p","tpsa_p","qed_p"])
    props = pd.concat([valids, props], axis=1)
    
    smi = pd.DataFrame(data={'smiles': smiles})
    smi = pd.concat([smi, props], axis=1)

    return props


# def property_from_smiles(smiles, props_predictor, n_jobs=1):
#     """
#     Compute validity and properties of a
#     list of smiles and return a pd.DataFrame
#     """
#     with Pool(n_jobs) as pool:
#         props = pool.map(props_predictor, smiles)
#     valids, props = map(lambda x: np.asarray(x), zip(*props))
#     valids = pd.DataFrame(data={'valid': valids})
#     props = pd.DataFrame(props, columns=["logp_p","tpsa_p","qed_p"])
#     props = pd.concat([valids, props], axis=1)
    
#     smi = pd.DataFrame(data={'smiles': smiles})
#     smi = pd.concat([smi, props], axis=1)

#     return props


# def props_predictor_wrapper(predictor, conditions):
#     def props_predictor(smiles):
#         mol = MolFromSmiles(smiles)
#         if mol is not None:
#             valid = 1
#             props = [predictor[c](mol) for c in conditions]
#         else:
#             valid = 0
#             props = [np.nan]*len(conditions)
#         return valid, props
#     return props_predictor


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

        if len(smis_groups[i]) == 0:
            not_intersected.append(0)    
        elif len(adjacent_smis) == 0:
            not_intersected.append(1)
        else:
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


def get_metrics_from_smiles_file(save_folder, n_steps,
                                 train_smiles, n_jobs, prefix):
    metrics, smis_groups, mols_groups = [], [], []
    for i in range(n_steps+1):
        save_path = os.path.join(save_folder, f'{prefix}_{i}.csv')
        smiles = pd.read_csv(save_path, index_col=[0])
        smiles = smiles["smiles"].dropna().tolist()
        
        m, smis, mols = compute_metrics(smiles, train_smiles, n_jobs)
        metrics.append(m)
        smis_groups.append(smis)
        mols_groups.append(mols)

    snn_start, snn_prev = snn_of_mol_groups(mols_groups)
    not_intersected = percent_not_intersected(smis_groups)

    df = pd.DataFrame({
        "validity": [m['valid'] for m in metrics],
        "uniqueness": [m['unique'] for m in metrics],
        "novelty": [m['novel'] for m in metrics],
        "internal_diversity": [m['intDiv'] for m in metrics],
        "snn_start": snn_start,
        "snn_previous": snn_prev,
        "percent_not_intersect": not_intersected
    })
    return df


def validate_smiles(smiles, props_predictor):
    valid, src_props = props_predictor(smiles)
    if valid == 0:
        exit(f"SMILES is not valid: {smiles}")
    logp, tpsa, qed = src_props
    return [logp, tpsa, qed]


def create_src_file(smiles, props_predictor,
                    trg_props, save_path, scaler):
    src_props = list(validate_smiles(smiles, props_predictor))
    src_props = scaler.transform([src_props])[0]
    trg_props = scaler.transform([trg_props])[0]
    
    df = pd.DataFrame({
        "src": smiles,
        "src_logP": [src_props[0]],
        "src_tPSA": [src_props[1]],
        "src_QED": [src_props[2]],
        "trg_logP": [trg_props[0]],
        "trg_tPSA": [trg_props[1]],
        "trg_QED": [trg_props[2]]
    })
    df.to_csv(save_path, index=False)


def continuity_check_given_src(generator, smiles, trg_props, conditions,
                               storage_path, toklen, n_samples, n_jobs,
                               train_smiles, fields, std=0.2, logger=None):
    save_folder = os.path.join(storage_path, f"toklen{toklen}_src")
    os.makedirs(save_folder, exist_ok=True)
    props_predictor = props_predictor_wrapper(conditions)

    LOG = logger('continuity_check_given_src', log_path=os.path.join(save_folder, "records.log"))

    create_src_file(smiles, props_predictor, trg_props,
                    os.path.join(save_folder, "src.csv"),
                    generator.scaler)

    dataset = data.TabularDataset(path=os.path.join(save_folder, 'src.csv'),
                                  format='csv', fields=fields, skip_header=True)
    data_iter = data.BucketIterator(dataset, batch_size=1)
    
    dataloader = to_dataloader(data_iter, conditions, generator.pad_id,
                               generator.max_strlen, generator.device)

    for i, batch in enumerate(dataloader):
        z = generator.sample_z_from_src(batch.src, batch.econds)[0]
        for i in range(n_samples):
            toklen = int(np.random.normal(36, 3, 1)[0])
            _z = z[:, :toklen, :]

            z2 = _z + torch.empty_like(_z).normal_(mean=0, std=std)
            smiles = generator.sample_smiles(batch.dconds, z2,
                                             transform=False)[0]
            print(smiles[0])


def prepare_consecutive_props(n, props_peak, varId, varBound, alpha=0.80):
    var_prop_dist = (varBound[1] - varBound[0]) * alpha
    var_prop_begin = props_peak[varId] - var_prop_dist/2
    step_dist = var_prop_dist / n

    all_props = np.tile(props_peak, (n+1,1))
    all_props[:, varId] = [var_prop_begin + step_dist * i for i in range(n+1)]
    return all_props


def generate_on_var_conds(generator, n, z, props, var_prop_name, save_folder, bound, n_jobs, LOG):
    for i, props_t in enumerate(props):
        save_path = os.path.join(save_folder, f"{var_prop_name}_{i}.csv")

        props_t = augment_props(n, props_t, bound, 0, 0.6)
        smiles, *_ = generator.sample_smiles(torch.Tensor(props_t),
                                             augment_z(n, z),
                                             transform=True)
        LOG.info(smiles[:5])
        
        with Pool(n_jobs) as pool:
            props_p = np.array(pool.map(predict_props, smiles))
        valids = [0 if np.nan in p else 1 for p in props_p]

        smiles_props = pd.DataFrame({
            "smiles": smiles, "logp_t": props_t[:, 0],
                              "tpsa_t": props_t[:, 1],
                              "qed_t":  props_t[:, 2],
            "valids": valids, "logp_p": props_p[:, 0],
                              "tpsa_p": props_p[:, 1],
                              "qed_p":  props_p[:, 2]            
        })
        smiles_props.to_csv(save_path)
    
    
class ContinuityCheck:
    def __init__(self, args, generator, train_smiles, LOG):
        self.LOG = LOG
        self.n_jobs = args.n_jobs
        self.toklen = args.toklen
        self.latent_dim = args.latent_dim
        self.generator = generator
        self.train_smiles = train_smiles

        self.props = args.conditions
        self.props_peak = args.properties
        self.props_bound = { "logP": [args.logp_lb, args.logp_ub],
                             "tPSA": [args.tpsa_lb, args.tpsa_ub],
                             "QED" : [args.qed_lb,  args.qed_ub ]
                             }
        self.props_std = { "logP": 0.15,
                           "tPSA": 2.50,
                           "QED" : 0.05 
                           }

        self.n_steps = args.n_steps
        self.n_samples = args.n_samples
        print("init:", self.n_steps, self.n_samples)

    def _snn_start(self, mol_groups):
        snn_start = np.ones((len(mol_groups),), dtype=np.float32)
        for i in range(1, len(mol_groups)):
            snn_start[i] = get_snn_from_mol(mol_groups[0], mol_groups[i])
        return snn_start
        
    def _snn_previous(self, mol_groups):
        snn_prev = np.full((len(mol_groups),), np.nan, dtype=np.float32)
        for i in range(1, len(mol_groups)):
            snn_prev[i] = get_snn_from_mol(mol_groups[i-1], mol_groups[i])
        return snn_prev
    
    def _not_intersected(self, smi_groups):
        not_intersected = []
        for i in range(len(smi_groups)):
            if i == 0:
                adj_smi = set(smi_groups[i+1])
            elif i == len(smi_groups)-1:
                adj_smi = set(smi_groups[i-1])
            else:
                adj_smi = set(smi_groups[i-1]) | set(smi_groups[i+1])
                
            intersected_part = set(smi_groups[i]).intersection(adj_smi)

            if len(smi_groups[i]) == 0:
                not_intersected.append(0)    
            elif len(adj_smi) == 0:
                not_intersected.append(1)
            else:
                not_intersected.append(1-len(intersected_part)/len(adj_smi))
        return not_intersected
    
    def _rand_z(self, n, toklen, latent_dim):
        return torch.Tensor(np.random.normal(
            size=(n, toklen, latent_dim)))
        
    def _get_z(self, toklen, latent_dim, file_path):
        if not os.path.exists(file_path):
            z = self._rand_z(1, toklen, latent_dim)
            torch.save(z, file_path)
        z = torch.load(file_path)
        return z
        
    def _augment_props(self, n, props, var_prop=None):
        props = np.array(props).reshape((1, 3))
        props = np.repeat(props, n, axis=0)
        if var_prop is None:
            return props

        varid = self.props.index(var_prop)
        var_bound = self.props_bound[var_prop]
        var_std = self.props_std[var_prop]
        
        props[:, varid] += np.random.normal(0, var_std, (n,))
        
        for i in range(n):
            props[i, varid] = min(props[i, varid], var_bound[1])
            props[i, varid] = max(props[i, varid], var_bound[0])
        return props

    def _augment_z(self, n, z, std=None):
        # z: torch.Tensor
        assert z.dim() == 3
        z = z.repeat(n, 1, 1)
        if std:
            z += torch.empty_like(z).normal_(mean=0, std=std)
            return z
        return z
    
    def _save_props(self, smiles, props_t, save_path):
        with Pool(self.n_jobs) as pool:
            props_p = np.array(pool.map(predict_props, smiles))
        valids = [0 if np.nan in p else 1 for p in props_p]
        smiles_props = pd.DataFrame({
            "smiles": smiles, "logp_t": props_t[:, 0],
                              "tpsa_t": props_t[:, 1],
                              "qed_t" : props_t[:, 2],
            "valids": valids, "logp_p": props_p[:, 0],
                              "tpsa_p": props_p[:, 1],
                              "qed_p" : props_p[:, 2]
        })
        smiles_props.to_csv(save_path)

    def calc_statistics(self, save_folder, n_steps, file_name):
        metrics, smis_groups, mols_groups = [], [], []
        
        for i in range(n_steps+1):
            smiles = pd.read_csv(os.path.join(save_folder, 
                     f'{file_name}_{i}.csv'), index_col=[0])
            
            smiles = smiles["smiles"].dropna().tolist()
            
            with Pool(self.n_jobs) as pool:
                mols = pool.map(get_mol, smiles)
                mols = [mol for mol in mols if mol]
                smis = pool.map(get_smiles, mols) # canonical
            
            m = get_basic_metrics(smiles, self.train_smiles, self.n_jobs)

            smis_groups.append(smis)
            mols_groups.append(mols)
            metrics.append({ "valid":  m["valid"],
                             "unique": m["unique"],
                             "novel":  m["novel"],
                             "intDiv": m["intDiv"]
                           })

        df = pd.DataFrame({
            "validity":          [m['valid'] for m in metrics],
            "uniqueness":        [m['unique'] for m in metrics],
            "novelty":           [m['novel'] for m in metrics],
            "int_div":           [m['intDiv'] for m in metrics],
            "snn_start":         self._snn_start(mols_groups),
            "snn_previous":      self._snn_previous(mols_groups),
            "not_intersect":     self._not_intersected(smis_groups)
        })
        df.to_csv(os.path.join(save_folder, f"{file_name}_statistics.csv"))

    def calc_errors(self, save_folder, prop_name):
        save_path = os.path.join(save_folder, f'{prop_name}_error.csv')
        
        with open(save_path, 'w') as ptr:
            for i in range(self.n_steps+1):
                gen = pd.read_csv(os.path.join(save_folder, 
                                  f"{prop_name}_{i}.csv"))
                metrics = get_all_metrics(gen, self.train_smiles, self.n_jobs)
                header, body = print_all_metrics(metrics)
                
                if i == 0:
                    ptr.write(header+'\n')
                ptr.write(body+'\n')


class ContinuityCheckOnConds(ContinuityCheck):
    def __init__(self, args, generator, train_smiles, LOG):
        super().__init__(args, generator, train_smiles, LOG)
    
    def _get_consecutive_props(self, var_prop, alpha=0.80):
        var_id = self.props.index(var_prop)
        bound = self.props_bound[var_prop]
        
        var_prop_dist = (bound[1] - bound[0]) * alpha
        var_prop_begin = self.props_peak[var_id] - var_prop_dist/2
        step_dist = var_prop_dist / self.n_steps

        all_props = np.tile(self.props_peak, (self.n_steps+1,1))
        for i in range(self.n_steps+1):
            all_props[i, var_id] = var_prop_begin + step_dist * i
        return all_props
    
    def _generate_on_var_props(self, z, props, var_prop, save_folder):
        z = self._augment_z(self.n_samples, z)
        
        for i, props_t in enumerate(props):
            props_t = self._augment_props(self.n_samples, props_t, var_prop)
            smiles, *_ = self.generator.sample_smiles(torch.Tensor(props_t),
                                                      z, transform=True)
            self.LOG.info(smiles[:4])
            self._save_props(smiles, props_t, os.path.join(
                             save_folder, f"{var_prop}_{i}.csv"))
            
            
    def generate(self, save_folder):
        z = self._get_z(self.toklen, self.latent_dim, 
                        os.path.join(save_folder, 'z.pt'))

        for p in self.props:
            props_t = self._get_consecutive_props(p)
            self._generate_on_var_props(z, props_t, p, save_folder)

            self.LOG.info(f"calculate statistics on {p}...")
            self.calc_statistics(save_folder, self.n_steps, p)

            self.LOG.info(f"calculate errors on {p}...")
            self.calc_errors(save_folder, p)


class ContinuityCheckOnZ(ContinuityCheck):
    """Continuity check for the latent space:
    Check if the latent space is continuous by sampling
    from several consecutive equal-spacing points between
    two latent space.
    """
    def __init__(self, args, generator, train_smiles, LOG):
        super().__init__(args, generator, train_smiles, LOG)
    
    def _distance(self, z1, z2):
        return torch.sqrt(torch.sum((z2 - z1)**2)).item()
    
    def _get_std(self, z_dist_each, alpha=0.5, upperbound=1):
        std = (z_dist_each / 2) * alpha
        std = std if std < upperbound else upperbound
        return std

    def _get_consecutive_zs(self, save_folder):
        z1 = self._get_z(self.toklen, self.latent_dim, 
                         os.path.join(save_folder, 'z1.pt'))
        z2 = self._get_z(self.toklen, self.latent_dim, 
                         os.path.join(save_folder, 'z2.pt'))
        z_vec = z2 - z1
        z_dist_each = self._distance(z1, z2) / self.n_steps
        
        zs = []
        for i in range(self.n_steps+1):
            zs.append(z1 + (z_vec / self.n_steps) * i)
        return zs, z_dist_each

    def generate(self, save_folder):
        zs, z_dist_each = self._get_consecutive_zs(save_folder)

        props_t = self._augment_props(self.n_samples, self.props_peak)
        props_in = torch.Tensor(props_t)
        # props_df = pd.DataFrame(props_t, columns=["logp_t","tpsa_t","qed_t"])
        
        zstd = self._get_std(z_dist_each)
        self.LOG.info(f"std: {zstd}")
        
        for i in range(self.n_steps + 1):
            self.LOG.info(f"sample smiles {i} / {self.n_steps}")
            save_path = os.path.join(save_folder, f"z1z2_{i}.csv")
            # if os.path.exists(save_path):
            #     continue

            aug_z = self._augment_z(self.n_samples, zs[i], zstd)
            smiles, *_ = self.generator.sample_smiles(props_in, aug_z,
                                                      transform=True)
            self.LOG.info(smiles[:4])
            self.LOG.info("save properties...")
            self._save_props(smiles, props_in, save_path)
            
        self.LOG.info("calculate statistics...")
        self.calc_statistics(save_folder, self.n_steps, 'z1z2')
        
        self.LOG.info("calculate errors...")
        self.calc_errors(save_folder, 'z1z2')


def continuity_check(args, generator, train_smiles, logger):
    save_folder = os.path.join(args.storage_path,
                               f"check_{args.test_for}",
                               f"toklen{args.toklen}",
                               args.decode_algo)
    os.makedirs(save_folder, exist_ok=True)
 
    LOG = logger(f'continuity check on {args.test_for}',
                 log_path=os.path.join(save_folder, "records.log"))

    if args.test_for == "z":
        ccoz = ContinuityCheckOnZ(args, generator, train_smiles, LOG)
        ccoz.generate(save_folder)
        
    elif args.test_for == "conds":
        ccoc = ContinuityCheckOnConds(args, generator, train_smiles, LOG)
        ccoc.generate(save_folder)


def continuity_check_on_z(args, generator, train_smiles, save_folder, LOG):
    """Continuity check for the latent space:
    Check if the latent space is continuous by sampling
    from several consecutive equal-spacing points between
    two latent space.

    validities: the validity of each smiles group   
    intersections: the percentage of smiles not in the adjacent smiles group(s)
    snn_start: snn between the first smiles group and the other ones. 
    snn_prev: snn between two consecutive smiles groups
    """

    zs = get_z_points(args.toklen, args.latent_dim, save_folder)
    zs_vec = zs[1] - zs[0]
    zs_dist = distance(zs[1], zs[0])
    LOG.info(f"Distance between 2 zs: {zs_dist}")

    calc_props = props_predictor_wrapper(property_prediction, args.conditions)
    
    props_t = augment_props(args.n_samples, args.properties)
    props_t = pd.DataFrame(props_t, columns=["logp_t","tpsa_t","qed_t"])

    std = (zs_dist / args.n_steps * 0.5) * 0.5
    std = std if std < 1 else 1
    LOG.info(f"std: {std}")

    assert (zs[0] + zs_vec == zs[1]).any()
    
    for i in range(args.n_steps+1):
        LOG.info(f"sample smiles {i} / {args.n_steps}")
        save_path = os.path.join(save_folder, f"z1z2_{i}.csv")
        if os.path.exists(save_path):
            continue
        
        z = zs[0] + (zs_vec / args.n_steps) * i
        smiles, _, _ = sample_smiles(z, args.properties, args.n_samples,
                                     generator, zstd=std)
        props_p = property_from_smiles(smiles, calc_props, args.n_jobs)
        smiles = pd.DataFrame(smiles, columns=['smiles'])
        smiles_props = pd.concat([smiles, props_t, props_p], axis=1)
        smiles_props.to_csv(save_path)
        LOG.info("check 5 SMILES:\n" + "\n".join(smiles['smiles'].iloc[:5]))
    
    if not os.path.exists(os.path.join(save_folder, "statistics.csv")):
        LOG.info("store metrics from smiles file...")
        dist = pd.DataFrame({
            "distance_start": [zs_dist/args.n_steps*i for i in range(args.n_steps+1)],
            "distance_previous": [np.nan] + [zs_dist/args.n_steps]*args.n_steps
        })
        df = get_metrics_from_smiles_file(save_folder, args.n_steps, train_smiles,
                                        args.n_jobs, prefix="z1z2")
        df = pd.concat([dist, df], axis=1)
        df.to_csv(os.path.join(save_folder, "statistics.csv"))
        LOG.info("execution finished.")

    LOG.info("compute error...")
    with open(os.path.join(save_folder, 'error.csv'), 'w') as ptr:
        for i in range(args.n_steps+1):
            data_path = os.path.join(save_folder, f'z1z2_{i}.csv')
            gen = pd.read_csv(data_path)
            all_metrics = get_all_metrics(gen, train_smiles, args.n_jobs)
            header, body = print_all_metrics(all_metrics)
            if i == 0:
                ptr.write(header+'\n')
            ptr.write(body+'\n')
            

def store_metrics_of_transvae_gen_smiles(zs_dist, n_steps, n_jobs):
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
    
    dist = pd.DataFrame({
        "distance_start": [zs_dist/n_steps*i for i in range(n_steps+1)],
        "distance_previous": [np.nan] + [zs_dist/n_steps]*n_steps
    })
    df = get_metrics_from_smiles_file(save_folder, n_steps,
        train_smiles, n_jobs, prefix="z1z2")
    df = pd.concat([dist, df], axis=1)
    df.to_csv(os.path.join(save_folder, "statistics.csv"))
    exit(0)
    
    
# def continuity_check_on_conds(args, generator, train_smiles, save_folder, LOG):
#     """Continuity check on the conditions
#     Check if the conditions is continuous by sampling...
#     """

#     z = get_z_points(args.toklen, args.latent_dim, save_folder, ('z.pt',))[0]
#     logp_peak, tpsa_peak, qed_peak = args.properties

#     logp_dist = (args.logp_ub - args.logp_lb) * 0.80
#     tpsa_dist = (args.tpsa_ub - args.tpsa_lb) * 0.80
#     qed_dist = (args.qed_ub - args.qed_lb) * 0.80

#     logp1 = logp_peak - logp_dist/2
#     tpsa1 = tpsa_peak - tpsa_dist/2
#     qed1 = qed_peak - qed_dist/2
    
#     bound = { "logp": [args.logp_lb, args.logp_ub],
#               "tpsa": [args.tpsa_lb, args.tpsa_ub],
#               "qed":  [args.qed_lb, args.qed_ub] }
    
#     p = prepare_consecutive_props(args.n_samples, args.properties, 0, bound['logp'])

#     generate_on_var_conds(generator, args.n_samples, z, p, 'logp', save_folder,bound['logp'], args.n_jobs, LOG)
    

#     # test logp difference
#     def generate(prop_name, pstd, propId):
#         all_props = []
        
#         for i in range(args.n_steps+1):
#             LOG.info(f"sample smiles {i} / {args.n_steps}")
#             save_path = os.path.join(save_folder, f"{prop_name}_{i}.csv")
#             if os.path.exists(save_path):
#                 continue

#             if prop_name == "logp":
#                 logpi = logp1 + (logp_dist / args.n_steps) * i
#                 props = [logpi, tpsa_peak, qed_peak]
#             elif prop_name == "tpsa":
#                 tpsai = tpsa1 + (tpsa_dist / args.n_steps) * i
#                 props = [logp_peak, tpsai, qed_peak]
#             elif prop_name == "qed":
#                 qedi = qed1 + (qed_dist / args.n_steps) * i
#                 props = [logp_peak, tpsa_peak, qedi]

#             all_props.append(props)
#             LOG.info(props)

#             props = augment_props(args.n_samples, props,
#                                   bound[prop_name], propId, pstd)
#             # pstds = [0, 0, 0]
#             # pstds[propId] = pstd
#             # props = augment_props(args.n_samples, props, bound,
#             #                       [propId], pstds)
#             props_t = torch.Tensor(props)
#             z_all = augment_z(args.n_samples, z)
            
#             smiles = generator.sample_smiles(props_t, z_all, transform=True)[0]
#             LOG.info(smiles[:5])
            
#             smiles = pd.DataFrame(smiles, columns=['smiles'])
#             props_t = pd.DataFrame(props_t, columns=["logp_t","tpsa_t","qed_t"])
#             props_p = property_from_smiles(smiles["smiles"], args.n_jobs)
            
#             # with Pool(args.n_jobs) as pool:
#             #     props = pool.map(predict_props, smiles)
#             # valids = [0 if np.nan in p else 1 for p in props]


#             smiles_props = pd.concat([smiles, props_t, props_p], axis=1)
#             smiles_props.to_csv(save_path)

#         df = get_metrics_from_smiles_file(save_folder, args.n_steps, 
#             train_smiles, args.n_jobs, prefix=prop_name)
#         df.to_csv(os.path.join(save_folder, f"{prop_name}_statistics.csv"))

#         LOG.info("compute error...")
#         with open(os.path.join(save_folder, f'{prop_name}_error.csv'), 'w') as ptr:
#             for i in range(args.n_steps+1):
#                 data_path = os.path.join(save_folder, f"{prop_name}_{i}.csv")
#                 gen = pd.read_csv(data_path)
#                 all_metrics = get_all_metrics(gen, train_smiles, args.n_jobs)
#                 header, body = print_all_metrics(all_metrics)
#                 print(header)
#                 print(body)
#                 if i == 0:
#                     ptr.write(header+'\n')
#                 ptr.write(body+'\n')

#     logp_std = 0.15
#     tpsa_std = 2.50
#     qed_std = 0.05

#     LOG.info(f"std: logP={logp_std}, tPSA={tpsa_std}, QED={qed_std}")

#     generate("logp", logp_std, 0)
#     generate("tpsa", tpsa_std, 1)
#     generate("qed", qed_std, 2)