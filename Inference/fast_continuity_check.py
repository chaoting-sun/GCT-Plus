import os
import numpy as np
import pandas as pd
import torch
from torchtext import data
from rdkit.Chem import MolFromSmiles
from moses.metrics import SNNMetric
from pathos.multiprocessing import ProcessingPool as Pool

from Utils.smiles import smi_to_mol, mol_to_smi
from Inference.metrics import get_all_metrics, get_snn_from_mol, get_basic_metrics, print_all_metrics
# from Utils.properties import predict_props
from Utils.dataset import to_dataloader
from Model.build_model import get_model
from Inference.model_prediction import Predictor
from Inference.utils import prepare_generator


class ContinuityCheck:
    def __init__(self, args, generator, train_smiles, LOG):
        self.LOG          = LOG
        self.n_jobs       = args.n_jobs
        self.toklen       = args.toklen
        self.latent_dim   = args.latent_dim
        self.generator    = generator
        self.train_smiles = train_smiles

        self.props        = args.conditions
        self.props_peak   = args.properties
        self.props_bound  = { "logP": [args.logp_lb, args.logp_ub],
                              "tPSA": [args.tpsa_lb, args.tpsa_ub],
                              "QED" : [args.qed_lb,  args.qed_ub ]
                             }
        self.props_std    = { "logP": 0.15,
                              "tPSA": 2.50,
                              "QED" : 0.05 
                            }

        self.n_steps      = args.n_steps
        self.n_samples    = args.n_samples
        
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
        
    def _augment_props(self, n, props, var_prop=None):
        props = np.array(props).reshape((1, 3))
        props = np.repeat(props, n, axis=0)
        if var_prop is None:
            return props
        if len(props) == 1:
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
                mols = pool.map(smi_to_mol, smiles)
                mols = [mol for mol in mols if mol]
                smis = pool.map(mol_to_smi, mols) # canonical
            
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
            
            
    def generate(self, save_folder, z):
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

    def _get_consecutive_zs(self, z1, z2):
        z_vec = z2 - z1
        z_dist_each = self._distance(z1, z2) / self.n_steps
        
        zs = []
        for i in range(self.n_steps+1):
            zs.append(z1 + (z_vec / self.n_steps) * i)
        return zs, z_dist_each

    def generate(self, save_folder, z1, z2):
        zs, z_dist_each = self._get_consecutive_zs(z1, z2)

        props_t = self._augment_props(self.n_samples, self.props_peak)
        props_in = torch.Tensor(props_t)
        # props_df = pd.DataFrame(props_t, columns=["logp_t","tpsa_t","qed_t"])
        
        zstd = self._get_std(z_dist_each)
        self.LOG.info(f"std: {zstd}")
        
        for i in range(self.n_steps + 1):
            self.LOG.info(f"sample smiles {i} / {self.n_steps}")
            save_path = os.path.join(save_folder, f"z_{i}.csv")
            # if os.path.exists(save_path):
            #     continue

            aug_z = self._augment_z(self.n_samples, zs[i], zstd)
            smiles, *_ = self.generator.sample_smiles(props_in, aug_z,
                                                      transform=True)
            self.LOG.info(smiles[:4])
            self.LOG.info("save properties...")
            self._save_props(smiles, props_in, save_path)
            
        self.LOG.info("calculate statistics...")
        self.calc_statistics(save_folder, self.n_steps, 'z')
        
        self.LOG.info("calculate errors...")
        self.calc_errors(save_folder, 'z')


def build_save_path(args):
    save_folder = os.path.join(args.storage_path,
                               f"check_{args.test_for}",
                               f"toklen{args.toklen}",
                               args.decode_algo)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


# def plot_metric():


def continuity_check(args, generator, train_smiles, logger):
    save_folder = build_save_path(args)

    LOG = logger(f'continuity check on {args.test_for}',
                 log_path=os.path.join(save_folder, "records.log"))

    if args.test_for == "z":
        ccoz = ContinuityCheckOnZ(args, generator, train_smiles, LOG)
        ccoz.generate(save_folder)
        
    elif args.test_for == "conds":
        ccoc = ContinuityCheckOnConds(args, generator, train_smiles, LOG)
        ccoc.generate(save_folder)


def random_z(n, toklen, latent_dim):
    return torch.Tensor(np.random.normal(size=(n, toklen, latent_dim)))


def prepare_random_z(toklen, latent_dim, save_path):
    if not os.path.exists(save_path):
        z = random_z(1, toklen, latent_dim)
        torch.save(z, save_path)
        return z
    return torch.load(save_path)


def fast_continuity_check(args, toklen_data, train_smiles, 
                          scaler, SRC, TRG, device, logger):
    os.makedirs(args.inference_path, exist_ok=True)
    os.makedirs(os.path.join(args.inference_path, 'check_z'), exist_ok=True)
    os.makedirs(os.path.join(args.inference_path, 'check_conds'), exist_ok=True)

    z1 = prepare_random_z(args.toklen, args.latent_dim,
                          os.path.join(args.inference_path, 'check_z', 'z1.pt'))
    z2 = prepare_random_z(args.toklen, args.latent_dim,
                          os.path.join(args.inference_path, 'check_z', 'z2.pt'))

    torch.sqrt(torch.sum((z2 - z1)**2)).item()

    z = prepare_random_z(args.toklen, args.latent_dim,
                         os.path.join(args.inference_path, 'check_conds', 'z.pt'))

    for epoch in args.epoch_list:
        """ 
        prepare generator
        """
        args.use_model_path = os.path.join(args.train_path, args.model_name, f'model_{epoch}.pt')
        generator = prepare_generator(args, SRC, TRG, toklen_data, scaler, device)

        """
        check continuity on Z
        """
        save_folder = os.path.join(args.inference_path, 'check_z', args.model_name, str(epoch))
        os.makedirs(save_folder, exist_ok=True)
        LOG = logger(name='continuity check on Z',
                     log_path=os.path.join(save_folder, "records.log"))

        ccoz = ContinuityCheckOnZ(args, generator, train_smiles, LOG)
        ccoz.generate(save_folder, z1, z2)

        """
        check continuity on properties
        """
        save_folder = os.path.join(args.inference_path, 'check_conds', args.model_name, str(epoch))
        os.makedirs(save_folder, exist_ok=True)

        LOG = logger(name='continuity check on properties',
                     log_path=os.path.join(save_folder, "records.log"))

        ccoc = ContinuityCheckOnConds(args, generator, train_smiles, LOG)
        ccoc.generate(save_folder, z)

        # del generator
