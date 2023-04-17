import torch
import numpy as np
from time import time
import torch.nn.functional as F
from Model.modules import get_trg_mask, get_src_mask
from Inference.toklen_sampling import tokenlen_gen_from_data_distribution
# from Inference.inputWrapping import InputWrapping
from Utils.mapper import mapper


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

        self.model        = model
        self.top_k        = top_k
    
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

    def sample_toklen(self, n):
        """sample n token lenths from training data
        
        Args:
            n (int): number of token lengths
        
        Returns:
            toklens (List[int]): a list of token lengths
        """
        n_bin = int(self.toklen_data.max()
                  - self.toklen_data.min())
        toklens = [
            int(tokenlen_gen_from_data_distribution(
            data=self.toklen_data, size=1,
            nBins=n_bin)) + self.cond_dim
            for _ in range(n)
        ]
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
                kws['trg_mask'] = get_trg_mask(ys, self.pad_id, self.use_cond2dec)
                if self.cond_dim > 0:
                    kws['dconds'] = kwargs['dconds']

                # obtain decoder output
                output_mol = self.model.decode(**kws)
                if self.use_cond2dec:
                    output_mol = output_mol[:, 3:, :]
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
    
    def sample_smiles(self, n=None, zs=None, toklen=None):
        # latent input
        if n is not None and zs is None and toklen is None:
            toklen = self.sample_toklen(n)
            max_toklen = max(toklen)
            zs = self.sample_z(max_toklen, n)

        elif n is None and zs is not None and toklen is not None:
            if not torch.is_tensor(zs):
                zs = torch.Tensor(zs)
            n = zs.size(0)
            max_toklen = max(toklen)
            assert zs.size(1) == max_toklen
        
        else:
            exit('Not supported!')

        # decoder embedding input
        ys = self.init_y(n, add_sos=True)
        
        # mask
        src_mask = torch.zeros((n,1,max_toklen), dtype=torch.bool)  
        for i in range(n):
            src_mask[i, 0, :toklen[i]] = True

        # move to gpu
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        src_mask = src_mask.to(self.device)
        
        # sample smiles
        outs = self.decode(zs=zs, ys=ys, src_mask=src_mask)
        outs = outs.cpu().numpy()
        smiles = [id_to_smi(ids, self.TRG) for ids in outs]
        toklen_gen = [len(smi) for smi in smiles]
        
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
    
    def sample_smiles(self, dconds, zs=None, transform=True):
        """sample SMILES from properties and scaffold smiles
        
        Args:
            prop (List[n, self.cond_dim]): target properties
            z (torch.FloatTensor): the latent space
        
        Returns:
            smiles (List[str]): list of SMILES string generated from the model
            toklen_gen (List[int]): list of token length of the generated SMILES
            toklen (List[int]): list of token length of the latent space
        """
        n = len(dconds)
                
        # prepare property input
        if transform:
            dconds = self.transform(dconds)
                    
        # sca_token = self.TRG.tokenize(sca_smi)
        
        # decoder embedding input
        ys = self.init_y(n, add_sos=True)

        # latent space input
        if zs is not None:
            assert torch.is_tensor(zs) is True
            smi_toklen = [zs.size(1)]*zs.size(0)
            lat_toklen = zs.size(1)
        else:    
            smi_toklen = self.sample_toklen(n)
            lat_toklen = max(smi_toklen)
            zs = self.sample_z(lat_toklen, n)
        
        # mask
        src_mask = torch.zeros((n,1,lat_toklen), dtype=torch.bool)  
        for i in range(n):
            src_mask[i, 0, :smi_toklen[i]] = True

        # move to gpu
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        dconds = dconds.to(self.device)
        src_mask = src_mask.to(self.device)
        
        # sample smiles
        outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [id_to_smi(ids, self.TRG) for ids in outs]
        toklen_gen = [len(smi) for smi in smiles]
        
        return smiles, smi_toklen, toklen_gen


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

    def sample_smiles(self, dconds, sca_smi, zs=None, transform=True):
        """sample SMILES from properties and scaffold smiles
        
        Args:
            prop (List[n, self.cond_dim]): target properties
            sca_smi (str): scaffold SMILES
            z (torch.FloatTensor): the latent space
        
        Returns:
            smiles (List[str]): list of SMILES string generated from the model
            toklen_gen (List[int]): list of token length of the generated SMILES
            toklen (List[int]): list of token length of the latent space
        """
        n = len(dconds)
                
        # prepare property input
        if transform:
            dconds = self.transform(dconds)

        src_ids = [self.TRG.vocab.stoi[e] for e
                   in self.TRG.tokenize(sca_smi)]

        # decoder embedding input
        ys = self.init_y(n, add_sos=True, sca_ids=src_ids, add_sep=True)

        # latent space input
        if zs is not None:
            assert torch.is_tensor(zs) is True
            smi_toklen = [zs.size(1)-len(src_ids)-1] * zs.size(0)
            lat_toklen = zs.size(1)
        else:    
            smi_toklen = self.sample_toklen(n)
            lat_toklen = len(src_ids) + max(smi_toklen) + 1
            zs = self.sample_z(lat_toklen, n)
        
        # mask
        src_mask = torch.zeros((n,1,lat_toklen), dtype=torch.bool)  
        for i in range(n):
            src_mask[i, 0, :len(src_ids)+1+smi_toklen[i]] = True

        # move to gpu
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        dconds = dconds.to(self.device)
        src_mask = src_mask.to(self.device)

        # sample smiles
        

        outs = self.decode(zs=zs, ys=ys, dconds=dconds, src_mask=src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [id_to_smi(ids[1+len(src_ids)+1:], self.TRG)
                  for ids in outs]
        toklen_gen = [len(smi) for smi in smiles]
        
        return smiles, smi_toklen, toklen_gen


# prob = self.predict(
#             trg=ys,
#             z=z,
#             src_mask=src_mask,
#             trg_mask=trg_mask
#         )


# prob = self.predict(
#     trg=ys,
#     z=z,
#     dconds=prop,
#     src_mask=src_mask,
#     trg_mask=trg_mask
# )


# prob = self.predict(
#     trg=ys,
#     z=z,
#     dconds=prop,
#     src_mask=src_mask,
#     trg_mask=trg_mask
# )

sampling_tools = {
    'vaetf'      : VaetfSampling,
    'cvaetf'     : CvaetfSampling,
    'ctf'        : CvaetfSampling,
    'scacvaetfv3': ScaCvaetfV3Sampling
}


        
# class ScaCvaetfV1Sampling(Sampling):
#     def __init__(self, model, kwargs):
#         super().__init__(model, kwargs)

#     def encode(self, batch, transform=True):
#         if transform:
#             batch['econds'] = self.transform(batch['econds'].cpu())

#         batch['src'] = batch['src'].to(self.device)
#         batch['econds'] = batch['econds'].to(self.device)

#         src_mask = get_src_mask(batch['src'],
#                                 self.TRG.vocab.stoi['<pad>'],
#                                 batch['econds'])
#         z, mu, log_var = self.model.encode(src=batch['src'],
#                                            econds=batch['econds'],
#                                            src_mask=src_mask)
#         return z, mu, log_var

#     def decode(self, z, ys, prop, src_mask):
#         break_condition = torch.zeros(z.size(0), dtype=torch.bool)

#         with torch.no_grad():
#             for i in range(self.max_strlen - 1):
#                 trg_mask = get_trg_mask(
#                     ys, self.TRG.vocab.stoi['<pad>'],
#                     self.use_cond2dec,prop)
#                 trg_mask = trg_mask.to(self.device)

#                 prob = self.predict(
#                     trg=ys,
#                     z=z,
#                     dconds=prop,
#                     src_mask=src_mask,
#                     trg_mask=trg_mask
#                 )

#                 prob = prob[:, -1, :]
#                 # prob: (bs, n_vocab)
                
#                 # prob = top_k_logits(prob, k=3)
                
#                 if self.decode_algo == 'greedy':
#                     _, next_word = torch.max(prob, dim=1)
#                     ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
                
#                 elif self.decode_algo == 'multinomial':
#                     next_word = torch.multinomial(prob, 1)
#                     ys = torch.cat([ys, next_word], dim=1)
#                     next_word = torch.squeeze(next_word)

#                 end_condition = (next_word.to('cpu') == self.TRG.vocab.stoi['<eos>'])
#                 break_condition = (break_condition | end_condition)

#                 if all(break_condition):
#                     break
#                 # break if all meet the condition
#         return ys

    
#     def sample_smiles(self, dconds, src_ids, zs=None, transform=True):
#         """sample SMILES from properties and scaffold smiles
        
#         Args:
#             prop (List[n, self.cond_dim]): target properties
#             sca_smi (str): scaffold SMILES
#             z (torch.FloatTensor): the latent space
        
#         Returns:
#             smiles (List[str]): list of SMILES string generated from the model
#             toklen_gen (List[int]): list of token length of the generated SMILES
#             toklen (List[int]): list of token length of the latent space
#         """
#         n = len(dconds)
                
#         # prepare property input
#         if transform:
#             dconds = self.transform(dconds)
#         # if transform:
#         #     prop = self.scaler.transform(prop)
#         # prop = torch.from_numpy(np.array(prop, dtype=np.float32))
        
#         # sca_token = self.TRG.tokenize(sca_smi)
        
#         # decoder embedding input
#         ys = self.init_y(n, add_sos=True, sca_ids=src_ids)

#         # latent space input
#         if zs is not None:
#             assert torch.is_tensor(zs) is True
#             smi_toklen = [zs.size(1) - len(src_ids)]*zs.size(0)
#             lat_toklen = zs.size(1)
#         else:    
#             smi_toklen = self.sample_toklen(n)
#             lat_toklen = len(src_ids) + max(smi_toklen)
#             zs = self.sample_z(lat_toklen, n)
        
#         # mask
#         src_mask = torch.zeros((n,1,lat_toklen), dtype=torch.bool)  
#         for i in range(n):
#             src_mask[i, 0, :len(src_ids)+smi_toklen[i]] = True

#         # move to gpu
#         ys = ys.to(self.device)
#         zs = zs.to(self.device)
#         dconds = dconds.to(self.device)
#         src_mask = src_mask.to(self.device)
        
#         # sample smiles
#         outs = self.decode(zs, ys, dconds, src_mask)
#         outs = outs.cpu().numpy()
        
#         smiles = [id_to_smi(ids[1+len(src_ids):], self.TRG)
#                   for ids in outs]
#         toklen_gen = [len(smi) for smi in smiles]
        
#         return smiles, smi_toklen, toklen_gen


# class ScaCvaetfV2Sampling(Sampling):
#     def __init__(self, model, kwargs):
#         super().__init__(model, kwargs)

#     def encode(self, batch, transform=True):
#         if transform:
#             batch['econds'] = self.transform(batch['econds'].cpu())

#         batch['src'] = batch['src'].to(self.device)
#         batch['econds'] = batch['econds'].to(self.device)
#         batch['src_scaffold'] = batch['src_scaffold'].to(self.device)

#         src_enc_mask = get_src_mask(torch.cat((batch['src_scaffold'],
#                                                batch['src']), 1),
#                                     self.TRG.vocab.stoi['<pad>'],
#                                     batch['econds'])
#         z, mu, log_var = self.model.encode(src=batch['src'],
#                                            src_scaffold=batch['src_scaffold'],
#                                            econds=batch['econds'],
#                                            src_mask=src_enc_mask)
#         return z, mu, log_var

#     @torch.no_grad()
#     def decode(self, zs, ys, prop, scaffold, src_enc_mask, src_dec_mask):
#         break_condition = torch.zeros(zs.size(0), dtype=torch.bool)

#         for i in range(self.max_strlen - 1):
#             trg_mask = get_trg_mask(
#                 ys, self.TRG.vocab.stoi['<pad>'],
#                 self.use_cond2dec, prop)
#             trg_mask = trg_mask.to(self.device)
            
#             prob = self.predict(
#                 trg=ys,
#                 trg_scaffold=scaffold,
#                 z=zs,
#                 dconds=prop,
#                 src_enc_mask=src_enc_mask,
#                 src_dec_mask=src_dec_mask,
#                 trg_mask=trg_mask
#             )
#             prob = prob[:, -1, :]
#             # prob: (bs, n_vocab)
            
#             # prob = top_k_logits(prob, k=3)

#             if self.decode_algo == 'greedy':
#                 _, next_word = torch.max(prob, dim=1)
#                 ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            
#             elif self.decode_algo == 'multinomial':
#                 next_word = torch.multinomial(prob, 1)
#                 ys = torch.cat([ys, next_word], dim=1)
#                 next_word = torch.squeeze(next_word)

#             end_condition = (next_word.to('cpu') == self.TRG.vocab.stoi['<eos>'])
#             break_condition = (break_condition | end_condition)
            
#             # print(i, self.max_strlen, break_condition)

#             if all(break_condition):
#                 break
#         return ys

#     def sample_smiles(self, dconds, src_ids, zs=None, transform=True):
#         """sample SMILES from properties and scaffold smiles
        
#         Args:
#             prop (List[n, self.cond_dim]): target properties
#             src_smi (str): scaffold SMILES
#             z (torch.FloatTensor): the latent space
        
#         Returns:
#             smiles (List[str]): list of SMILES string generated from the model
#             toklen_gen (List[int]): list of token length of the generated SMILES
#             toklen (List[int]): list of token length of the latent space
#         """
#         n = len(dconds)
                
#         # property input     
#         if transform:
#             dconds = self.transform(dconds)   

#         # src input
#         src_ids_with_se = ([self.TRG.vocab.stoi['<sos>']]
#                            +src_ids
#                           +[self.TRG.vocab.stoi['<eos>']])
#         src = torch.Tensor(src_ids_with_se).long().repeat(n,1)
        
#         # decoder embedding input
#         ys = self.init_y(n)

#         # latent space input
#         if zs is not None:
#             torch.is_tensor(zs) is True
#             toklen = [zs.size(1)-self.cond_dim-len(src_ids)]*zs.size(0)
#             lat_toklen = zs.size(1)
#         else:
#             zs = self.sample_z(lat_toklen, n)
#             toklen = self.sample_toklen(n)
#             lat_toklen = self.cond_dim + len(src_ids_with_se)-2 + max(toklen)
        
#         # mask input
#         src_enc_mask = torch.zeros((n, 1, lat_toklen)).bool()
#         for i in range(n):
#             src_enc_mask[i, 0, :self.cond_dim
#                                +len(src_ids_with_se)-2
#                                +toklen[i]] = True
#         # src_enc_mask (right): prop + sca + smi
        
#         src_dec_mask = torch.ones((n,1,self.cond_dim
#                                       +len(src_ids_with_se))
#                                   ).bool() # 27=3+22+2
#         # src_dec_mask (left): prop + <sos>smi<eos>
        
#         # move to gpu
#         ys = ys.to(self.device)
#         zs = zs.to(self.device)
#         src = src.to(self.device)
#         dconds = dconds.to(self.device)
#         src_enc_mask = src_enc_mask.to(self.device)
#         src_dec_mask = src_dec_mask.to(self.device)
        
#         # sample smiles
#         outs = self.decode(zs, ys, dconds, src, src_enc_mask, src_dec_mask)
#         outs = outs.cpu().numpy()
        
#         smiles = [id_to_smi(ids, self.TRG) for ids in outs]
#         toklen_gen = [len(smi) for smi in smiles]
#         return smiles, toklen, toklen_gen

