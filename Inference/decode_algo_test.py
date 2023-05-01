import torch
import numpy as np
from time import time
import torch.nn.functional as F
from Model.modules import get_trg_mask, get_src_mask
from Inference.toklen_sampling import tokenlen_gen_from_data_distribution
# from Inference.inputWrapping import InputWrapping
from Utils.mapper import mapper

torch.set_printoptions(threshold=10_000)


def id_to_smi(ids, TRG):
    """Convert ids into smiles based on TRG.
    
    Args:
        ids (List[int]): a list of integers representing a SMILES
        TRG (torchtext.data.Field): field for target

    Returns:
        smi (str): a string representing a molecule                 
    """
    smi = ''
    for i in ids:
        if i == TRG.vocab.stoi['<eos>']:
            break
        if i != TRG.vocab.stoi['<sos>']:
            smi += TRG.vocab.itos[i]
    return smi


def smi_to_id(smi, TRG, add_sos=False, add_sep=False, add_eos=False):
    """Convert smiles into ids based on TRG.
    
    Args:
        smi (str): a string representing a molecule
        TRG (torchtext.data.Field): field for target

    Returns:
        ids (List[int]): a list of integers representing a SMILES
    """
    token = TRG.tokenize(smi)
    ids = []
    if add_sos:
        ids.append(TRG.vocab.stoi['<sos>'])
    if add_sep:
        ids.append(TRG.vocab.stoi['<sep>'])
    ids.extend([TRG.vocab.stoi[t] for t in token])
    if add_eos:
        ids.append(TRG.vocab.stoi['<eos>'])    
    return ids


def top_k_logits(prob, k=4):
    v, ix = torch.topk(prob, k=k)
    out = prob.clone()
    out[out < v[:, [-1]]] = 1E-6
    return out


class Sampling:
    def __init__(self, model, kwargs, top_k=None):
        self.batch_size   = 512
        self.model        = model
        self.top_k        = top_k

        self.SRC          = kwargs['SRC']
        self.TRG          = kwargs['TRG']
        self.pad_id       = self.SRC.vocab.stoi['<pad>']
        self.sos_id       = self.TRG.vocab.stoi['<sos>']
        self.eos_id       = self.TRG.vocab.stoi['<eos>']
        self.sep_id       = self.TRG.vocab.stoi['<sep>']

        self.cond_dim     = kwargs['cond_dim']
        self.latent_dim   = kwargs['latent_dim']
        self.max_strlen   = kwargs['max_strlen']
        self.use_cond2dec = kwargs['use_cond2dec']
        self.decode_algo  = kwargs['decode_algo']
        self.toklen_data  = kwargs['toklen_data']
        
        self.scaler       = kwargs['scaler']
        self.device       = kwargs['device']
        self.n_jobs       = kwargs['n_jobs']
    
    def init_y(self, n, add_sos=True, sca_ids=None, add_sep=False):
        """create n starting sequences for prediction
        
        Args:
            n (int): number of starting sequences
            add_sos (bool): if start token is added
            sca_ids (List[int]): a list of ids
                representing the scaffold
        
        Returns:
            ys (torch.tensor): n starting sequences
                representated by ids
        """
        start_ids = []
        if add_sos:
            start_ids.append(self.sos_id)
        if sca_ids is not None:
            start_ids.extend(sca_ids)
        if add_sep:
            start_ids.append(self.sep_id)
        ys = torch.from_numpy(np.stack([start_ids]*n))
        return ys

    def id_to_smi(self, ids):
        """Convert ids into smiles based on TRG.
        
        Args:
            ids (List[int]): a list of integers representing a SMILES
            TRG (torchtext.data.Field): field for target

        Returns:
            smi (str): a string representing a molecule                 
        """
        smi = ''
        for i in ids:
            if i == self.TRG.vocab.stoi['<eos>']:
                break
            if i != self.TRG.vocab.stoi['<sos>']:
                smi += self.TRG.vocab.itos[i]
        return smi
    
    def smi_to_id(self, smi, add_sos=False, add_sep=False, add_eos=False):
        """Convert smiles into ids based on TRG.
        
        Args:
            smi (str): a string representing a molecule
            TRG (torchtext.data.Field): field for target

        Returns:
            ids (List[int]): a list of integers representing a SMILES
        """
        token = self.TRG.tokenize(smi)
        ids = []
        if add_sos:
            ids.append(self.TRG.vocab.stoi['<sos>'])
        if add_sep:
            ids.append(self.TRG.vocab.stoi['<sep>'])
        ids.extend([self.TRG.vocab.stoi[t] for t in token])
        if add_eos:
            ids.append(self.TRG.vocab.stoi['<eos>'])    
        return ids

    def sample_toklen(self, n):
        """sample n token lenths from training data
        
        Args:
            n (int): number of token lengths
        
        Returns:
            toklens (List[int]): a list of token lengths
        """
        n_bin = int(self.toklen_data.max()
                  - self.toklen_data.min())
        toklens = tokenlen_gen_from_data_distribution(data=self.toklen_data, size=n, nBins=n_bin)
        toklens = toklens.reshape((-1,)) + self.cond_dim
        toklens = (np.rint(toklens)).astype(int)
        return toklens

    def tokenize_smiles(self, smiles_list, field='SRC'):
        if field == 'SRC':
            p = self.SRC.process([self.SRC.tokenize(smi)
                                  for smi in smiles_list])
            if not self.SRC.batch_first:
                return p.T
        else:
            exit('no code!')
        return p
        
    def predict(self, **kwargs):
        output_mol = self.model.decode(**kwargs)
        if self.use_cond2dec:
            output_mol = output_mol[:, 3:, :]
        return F.softmax(output_mol, dim=-1)
    
    def sample_z(self, toklen, n):
        return torch.normal(
            mean=0, std=1,
            size=(n, toklen, self.latent_dim) 
        )
        
    def transform(self, prop):
        """rescale properties"""
        prop = self.scaler.transform(prop)
        return torch.from_numpy(prop).float()
    
    def encoder_input(
        self,
        smiles_list,
        transform=False,
        econds=None
    ):
        kwargs = {}
        # smiles
        kwargs['src'] = self.tokenize_smiles(smiles_list).to(self.device)
        # property
        if econds is not None:
            if transform:
                econds = self.transform(econds)
            if not torch.is_tensor(econds):
                econds = torch.from_numpy(np.array(econds))
            kwargs['econds'] = econds.to(self.device)
        return kwargs
    
    def decoder_input(
        self,
        ys,
        z,
        dconds=None,
        transform=False
    ):
        kwargs = {}
        kwargs['z'] = z.to(self.device)
        kwargs['trg'] = ys.to(self.device)        
        if dconds is not None:
            if transform:
                dconds = self.transform(dconds)
            if not torch.is_tensor(dconds):
                dconds = torch.from_numpy(np.array(dconds))
            kwargs['dconds'] = dconds.to(self.device)
        return kwargs

    def decode(self, **kwargs):
        ys = kwargs['ys']
        break_condition = torch.zeros(ys.size(0), dtype=torch.bool)
        
        with torch.no_grad():
            for i in range(self.max_strlen - 1):
                # prepare decoder input                
                kws = self.decoder_input(ys, kwargs['zs'])
                kws['src_mask'] = kwargs['src_mask']
                if self.cond_dim > 0:
                    kws['dconds'] = kwargs['dconds']
                    kws['trg_mask'] = get_trg_mask(ys, self.pad_id, self.use_cond2dec,
                                                   kwargs['dconds'])
                else:
                    kws['trg_mask'] = get_trg_mask(ys, self.pad_id, self.use_cond2dec)

                # obtain decoder output
                output_mol = self.model.decode(**kws)
                
                if self.use_cond2dec:
                    output_mol = output_mol[:, self.cond_dim:, :]
                
                prob = F.softmax(output_mol, dim=-1)
                prob = prob[:, -1, :] # (bs, n_vocab)
                
                # select top k values
                if self.top_k is not None:
                    prob = top_k_logits(prob, k=self.top_k)

                # select next word by certain algorithm
                if self.decode_algo == 'greedy':
                    _, next_word = torch.max(prob, dim=1)
                    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
                
                elif self.decode_algo == 'multinomial':
                    next_word = torch.multinomial(prob, 1)
                    ys = torch.cat([ys, next_word], dim=1)
                    next_word = torch.squeeze(next_word)

                # check breaking condition
                end_condition = (next_word.to('cpu') == self.eos_id)
                break_condition = (break_condition | end_condition)
                if all(break_condition):
                    break
        return ys


class VaetfSampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def encode_smiles(self, smiles_list):
        kwargs = self.encoder_input(smiles_list)
        kwargs['src_mask'] = get_src_mask(kwargs['src'], self.pad_id)
        z, mu, log_var = self.model.encode(**kwargs)
        return z, mu, log_var

    def encode_batch(self, batch):
        batch['src'] = batch['src'].to(self.device)
        src_mask = get_src_mask(batch['src'], self.pad_id)
        z, mu, log_var = self.model.encode(src=batch['src'],
                                           src_mask=src_mask)
        return z, mu, log_var
    
    def sample_smiles(self, n, zs=None, toklen=None):
        """unconditioned sampling

        Generally, Unconditioned sampling refers to two main purposes:
        1.  when a value for 'n' is given, the objective is to
        explore the chemical space in order to discover new and
        varied molecules that are expected to have similar 
        distributions to those in the training set
        2. when 'zs' (and 'toklen') are given, the aim is to
        obtain the SMILES representation for the latent space
        """

        ys = self.init_y(n, add_sos=True)

        if zs is not None:
            assert n == zs.size(0)

            if toklen is None:
                toklen = [zs.size(1)] * zs.size(0)
        else:
            if toklen is None:
                toklen = self.sample_toklen(n)

        max_toklen = max(toklen)
        
        if zs is None:
            zs = self.sample_z(max_toklen, n)

        # mask

        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        src_mask = torch.arange(max_toklen).expand(n,1,max_toklen) < toklen_stop_ids

        # print('toklen:', toklen)
        # print('src_mask:', src_mask)

        # move to gpu

        ys = ys.to(self.device)
        zs = zs.to(self.device)
        src_mask = src_mask.to(self.device)
        
        # sample smiles

        outs = self.decode(zs=zs, ys=ys, src_mask=src_mask)
        outs = outs.cpu().numpy()
        smiles = [id_to_smi(ids, self.TRG) for ids in outs]
        toklen_gen =[len(self.TRG.tokenize(smi)) for smi in smiles]
        
        return smiles, toklen, toklen_gen


class CvaetfSampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def encode_smiles(self, smiles_list, econds,
                      transform=True): 
        kwargs = self.encoder_input(smiles_list, transform, econds)
        kwargs['src_mask'] = get_src_mask(kwargs['src'], self.pad_id, kwargs['econds'])
        z, mu, log_var = self.model.encode(**kwargs)
        return z, mu, log_var

    def encode_batch(self, batch, transform=True):
        if transform:
            batch['econds'] = self.transform(batch['econds'].cpu())
        batch['src'] = batch['src'].to(self.device)
        batch['econds'] = batch['econds'].to(self.device)
        batch['src_mask'] = get_src_mask(batch['src'], self.pad_id, batch['econds']) 
        z, mu, log_var = self.model.encode(**batch)
        return z, mu, log_var

    def sample_smiles(self, dconds, zs=None, toklen=None, transform=True):
        if zs is not None:
            assert len(dconds) == len(zs), "The number of 'dconds' and 'zs' should be the same!"

        n = len(dconds)
        ys = self.init_y(n, add_sos=True)

        if transform:
            dconds = self.transform(dconds)

        if zs is not None:
            if toklen is None:
                toklen = [zs.size(1)] * zs.size(0)
        else:
            if toklen is None:
                toklen = self.sample_toklen(n)
            else:
                toklen = [t+self.cond_dim for t in toklen]
        
        max_toklen = max(toklen)

        if zs is None:
            zs = self.sample_z(max_toklen, n)

        # mask

        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        src_mask = torch.arange(max_toklen).expand(n,1,max_toklen) < toklen_stop_ids

        # move to gpu

        ys = ys.to(self.device)
        zs = zs.to(self.device)
        dconds = dconds.to(self.device)
        src_mask = src_mask.to(self.device)

        # sample smiles

        outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [id_to_smi(ids, self.TRG) for ids in outs]
        toklen_gen =[len(self.TRG.tokenize(smi)) for smi in smiles]
        
        return smiles, toklen, toklen_gen


class ScaCvaetfV1Sampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def encode_smiles(self, smiles_list, scaffold_list, econds,
                      transform=True):
        concat_smiles = [s1+'<sep>'+s2 for s1, s2 in
                         zip(smiles_list, scaffold_list)]
        kwargs = self.encoder_input(concat_smiles, transform, econds)
        kwargs['src_mask'] = get_src_mask(kwargs['src'], self.pad_id, kwargs['econds'])
        z, mu, log_var = self.model.encode(**kwargs)
        return z, mu, log_var

    def sample_smiles(self, dconds, scaffold, zs=None, toklen=None, transform=True):
        n = len(dconds)
                
        if transform:
            dconds = self.transform(dconds)

        sca_ids = [self.TRG.vocab.stoi[e] for e
                   in self.TRG.tokenize(scaffold)]

        ys = self.init_y(n, add_sos=True, sca_ids=sca_ids)

        if zs is not None:
            if toklen is None:
                toklen = [zs.size(1)-len(sca_ids)] * zs.size(0)
        else:
            if toklen is None:
                toklen = self.sample_toklen(n)

        max_toklen = max(toklen)
        lat_toklen = len(sca_ids) + max_toklen # +1 -> +<sep>

        if zs is None:
            zs = self.sample_z(lat_toklen, n)

        # mask
        
        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        toklen_stop_ids = torch.add(toklen_stop_ids, len(sca_ids))
        src_mask = torch.arange(lat_toklen).expand(n,1,lat_toklen) < toklen_stop_ids

        # move to gpu
        
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        dconds = dconds.to(self.device)
        src_mask = src_mask.to(self.device)

        # sample smiles

        outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [id_to_smi(ids[1+len(sca_ids):], self.TRG)
                  for ids in outs]
        toklen_gen =[len(self.TRG.tokenize(smi)) for smi in smiles]

        return smiles, toklen, toklen_gen
    
    
class ScaCvaetfV3Sampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def encode_smiles(self, smiles_list, scaffold_list, econds,
                      transform=True):
        concat_smiles = [s1+'<sep>'+s2 for s1, s2 in
                         zip(smiles_list, scaffold_list)]
        kwargs = self.encoder_input(concat_smiles, transform, econds)
        kwargs['src_mask'] = get_src_mask(kwargs['src'], self.pad_id, kwargs['econds'])
        z, mu, log_var = self.model.encode(**kwargs)
        return z, mu, log_var

    def sample_multiple_smiles1(self, dconds, scaffolds, zs=None, toklen=None, transform=True):
        sca_ids = np.array(mapper(smi_to_id, scaffolds, self.n_jobs))
        sca_ids_len = np.array(mapper(len, sca_ids, self.n_jobs))
        sort_index = np.argsort(sca_ids_len)
        sca_ids = sca_ids[sort_index]
        sca_ids_len = sca_ids_len[sort_index]

    def sample_multiple_smiles(self, dconds, scaffolds, transform=True):
        n = len(dconds)

        sca_ids = np.array([self.smi_to_id(sca) for sca in scaffolds])
        sca_lens = np.array(mapper(len, sca_ids, self.n_jobs))

        sort_indices = np.argsort(sca_lens)
        sca_ids = sca_ids[sort_indices]
        sca_lens = sca_lens[sort_indices]

        smiles, toklen, toklen_gen = [], [], []
        head_id = tail_id = 0

        # print('sca lens:', sca_lens)

        while head_id < n and tail_id <= n:
            if tail_id < n and sca_lens[head_id] == sca_lens[tail_id]:
                tail_id += 1
                continue
            
            print(f'# generated: {head_id} - {tail_id}')

            # print('head/tail id:', head_id, tail_id)
            # print('sca ids:', sca_ids[head_id:tail_id])
            
            cur_smiles, cur_toklen, cur_toklen_gen = self._sample_multiple_smiles(
                dconds[head_id:tail_id],
                sca_ids[head_id:tail_id],
                transform=transform
            )

            smiles.extend(cur_smiles)
            toklen.extend(cur_toklen)
            toklen_gen.extend(cur_toklen_gen)

            # print('cur_smiles:', cur_smiles)

            head_id = tail_id

        # print(smiles)

        return smiles, toklen, toklen_gen

    def _sample_multiple_smiles(self, dconds, sca_ids, transform=True):
        n = len(dconds)
                
        if transform:
            dconds = self.transform(dconds)

        sca_lens = mapper(len, sca_ids, self.n_jobs)

        max_sca_len = max(sca_lens)
        sca_del = [max_sca_len-l for l in sca_lens]
        
        ys = torch.ones((n, 1+max_sca_len+1), dtype=torch.long) * self.pad_id
        for i in range(n):
            ys[i, sca_del[i]] = self.sos_id
            ys[i, sca_del[i]+1:-1] = torch.tensor(sca_ids[i])
            ys[i, -1] = self.sep_id

        toklen = self.sample_toklen(n)

        src_toklen = [sca_lens[i]+1+toklen[i] for i in range(n)]
        lat_toklen = max(src_toklen)

        zs = self.sample_z(lat_toklen, n)

        toklen_stop_ids = torch.LongTensor(src_toklen).view(n, 1, 1)
        src_mask = torch.arange(lat_toklen).expand(n,1,lat_toklen) < toklen_stop_ids
        
        # move to gpu
        
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        dconds = dconds.to(self.device)
        src_mask = src_mask.to(self.device)
        
        # sample smiles

        outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
        outs = outs.cpu().numpy()

        smiles = [id_to_smi(ids[1+max_sca_len+1:], self.TRG)
                  for ids in outs]

        toklen_gen =[len(self.TRG.tokenize(smi)) for smi in smiles]

        return smiles, toklen, toklen_gen
    

    # def _sample_multiple_smiles(self, dconds, sca_ids, transform=True):
    #     n = len(dconds)
                
    #     if transform:
    #         dconds = self.transform(dconds)

    #     print('scaffolds:', scaffolds)

    #     sca_ids = [self.smi_to_id(sca) for sca in scaffolds]        
    #     sca_lens = mapper(len, sca_ids, self.n_jobs)

    #     print('sca_lens:', sca_lens)

    #     max_sca_len = max(sca_lens)
    #     sca_del = [max_sca_len-l for l in sca_lens]
        
    #     print('sca_del:', sca_del)
        
    #     ys = torch.ones((n, 1+max_sca_len+1), dtype=torch.long) * self.pad_id
    #     for i in range(n):
    #         ys[i, sca_del[i]] = self.sos_id
    #         ys[i, sca_del[i]+1:-1] = torch.tensor(sca_ids[i])
    #         ys[i, -1] = self.sep_id

    #     print('ys:', ys)

    #     if zs is not None:
    #         if toklen is None:
    #             toklen = [zs.size(1)-max_sca_len-1] * zs.size(0) # -<sep>
    #     else:
    #         if toklen is None:
    #             toklen = self.sample_toklen(n)
        
    #     print('toklen:', toklen)
        
    #     # mask

    #     src_toklen = [sca_lens[i]+1+toklen[i] for i in range(n)]
    #     lat_toklen = max(src_toklen)

    #     print('src_toklen:', src_toklen)

    #     if zs is None:
    #         zs = self.sample_z(lat_toklen, n)

    #     toklen_stop_ids = torch.LongTensor(src_toklen).view(n, 1, 1)
    #     src_mask = torch.arange(lat_toklen).expand(n,1,lat_toklen) < toklen_stop_ids

    #     print('src mask:', src_mask)
        
    #     # move to gpu
        
    #     ys = ys.to(self.device)
    #     zs = zs.to(self.device)
    #     dconds = dconds.to(self.device)
    #     src_mask = src_mask.to(self.device)
        
    #     # sample smiles

    #     outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
    #     outs = outs.cpu().numpy()
                
    #     print('outs:', outs)

    #     smiles = [id_to_smi(ids[1+max_sca_len+1:], self.TRG)
    #               for ids in outs]

    #     toklen_gen =[len(self.TRG.tokenize(smi)) for smi in smiles]

    #     return smiles, toklen, toklen_gen
    
    
    def sample_smiles(self, dconds, scaffold, zs=None, toklen=None, transform=True):
        n = len(dconds)
                
        if transform:
            dconds = self.transform(dconds)

        sca_ids = [self.TRG.vocab.stoi[e] for e
                   in self.TRG.tokenize(scaffold)]

        ys = self.init_y(n, add_sos=True, sca_ids=sca_ids, add_sep=True)

        if zs is not None:
            if toklen is None:
                toklen = [zs.size(1)-len(sca_ids)-1] * zs.size(0) # -<sep>
        else:
            if toklen is None:
                toklen = self.sample_toklen(n)
        
        max_toklen = max(toklen)
        lat_toklen = len(sca_ids) + 1 + max_toklen # +1 -> +<sep>

        if zs is None:
            zs = self.sample_z(lat_toklen, n)

        # mask
        
        toklen_stop_ids = torch.LongTensor(toklen).view(n, 1, 1)
        toklen_stop_ids = torch.add(toklen_stop_ids, len(sca_ids)+1)
        src_mask = torch.arange(lat_toklen).expand(n,1,lat_toklen) < toklen_stop_ids

        # move to gpu
        
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        dconds = dconds.to(self.device)
        src_mask = src_mask.to(self.device)

        # sample smiles

        outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [id_to_smi(ids[1+len(sca_ids)+1:], self.TRG)
                  for ids in outs]
        toklen_gen =[len(self.TRG.tokenize(smi)) for smi in smiles]

        return smiles, toklen, toklen_gen


sampling_tools = {
    'vaetf'      : VaetfSampling,
    'cvaetf'     : CvaetfSampling,
    'ctf'        : CvaetfSampling,
    'scacvaetfv1': ScaCvaetfV1Sampling,
    'scacvaetfv3': ScaCvaetfV3Sampling
}
