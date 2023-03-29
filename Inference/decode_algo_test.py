import torch
import numpy as np
from time import time
import torch.nn.functional as F
from Inference.toklen_sampling import tokenlen_gen_from_data_distribution
from Model.modules import create_target_mask, create_source_mask, nopeak_mask


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


def smi_to_id(smi, TRG, add_sos=False, add_eos=False):
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
    ids.extend([TRG.vocab.stoi[t] for t in token])
    if add_eos:
        ids.append(TRG.vocab.stoi['<eos>'])    
    return ids


class Sampling:
    def __init__(self, model, kwargs):
        self.model = model

        self.cond_dim     = kwargs['cond_dim']
        self.latent_dim   = kwargs['latent_dim']
        self.max_strlen   = kwargs['max_strlen']
        self.use_cond2dec = kwargs['use_cond2dec']
        self.decode_algo  = kwargs['decode_algo']
        self.toklen_data  = kwargs['toklen_data']
        
        self.scaler       = kwargs['scaler']
        self.device       = kwargs['device']
        self.TRG          = kwargs['TRG']
    
    def init_y(self, n, add_sos=True, sca_ids=None):
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
            start_ids.append(self.TRG.vocab.stoi['<sos>'])
        if sca_ids is not None:
            start_ids.extend(sca_ids)
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

    def encode(self, **kwargs):
        z, mu, log_var = self.model.encode(**kwargs)
        return z, mu, log_var

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
        
class ScaCvaetfV1Sampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    def decode(self, z, ys, prop, src_mask):
        break_condition = torch.zeros(z.size(0), dtype=torch.bool)

        with torch.no_grad():
            for i in range(self.max_strlen - 1):
                trg_mask = create_target_mask(
                    ys, self.TRG.vocab.stoi['<pad>'],
                    prop, self.use_cond2dec)
                trg_mask = trg_mask.to(self.device)

                prob = self.predict(
                    trg=ys,
                    z=z,
                    conds=prop,
                    src_mask=src_mask,
                    trg_mask=trg_mask
                )

                prob = prob[:, -1, :]
                # prob: (bs, n_vocab)
                
                if self.decode_algo == 'greedy':
                    _, next_word = torch.max(prob, dim=1)
                    ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
                
                elif self.decode_algo == 'multinomial':
                    next_word = torch.multinomial(prob, 1)
                    ys = torch.cat([ys, next_word], dim=1)
                    next_word = torch.squeeze(next_word)

                end_condition = (next_word.to('cpu') == self.TRG.vocab.stoi['<eos>'])
                break_condition = (break_condition | end_condition)

                if all(break_condition):
                    break
                # break if all meet the condition
        return ys

    def sample_smiles(self, prop, sca_smi, transform=True):
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
        n = len(prop)
                
        # prepare property input        
        if transform:
            prop = self.scaler.transform(prop)
        prop = torch.from_numpy(np.array(prop, dtype=np.float32))
        sca_ids = smi_to_id(sca_smi, self.TRG)
        # sca_token = self.TRG.tokenize(sca_smi)
        
        # decoder embedding input
        ys = self.init_y(n, add_sos=True, sca_ids=sca_ids)

        # decoder attention input
        smi_toklen = self.sample_toklen(n)
        lat_toklen = len(sca_ids) + max(smi_toklen)
        zs = self.sample_z(lat_toklen, n)
        src_mask = torch.zeros((n,1,lat_toklen), dtype=torch.bool)  
        for i in range(n):
            src_mask[i, 0, :len(sca_ids)+smi_toklen[i]] = True

        # move to gpu
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        prop = prop.to(self.device)
        src_mask = src_mask.to(self.device)
        
        # sample smiles
        outs = self.decode(zs, ys, prop, src_mask)
        outs = outs.cpu().numpy()
        
        smiles = [id_to_smi(ids[1+len(sca_ids):], self.TRG)
                  for ids in outs]
        toklen_gen = [len(smi) for smi in smiles]
        
        return smiles, smi_toklen, toklen_gen


class ScaCvaetfV2Sampling(Sampling):
    def __init__(self, model, kwargs):
        super().__init__(model, kwargs)

    @torch.no_grad()
    def decode(self, zs, ys, prop, scaffold, src_enc_mask, src_dec_mask):
        break_condition = torch.zeros(zs.size(0), dtype=torch.bool)

        for i in range(self.max_strlen - 1):
            trg_mask = create_target_mask(
                ys, self.TRG.vocab.stoi['<pad>'],
                prop, self.use_cond2dec)
            trg_mask = trg_mask.to(self.device)
            
            prob = self.predict(
                trg=ys,
                trg_scaffold=scaffold,
                z=zs,
                conds=prop,
                src_enc_mask=src_enc_mask,
                src_dec_mask=src_dec_mask,
                trg_mask=trg_mask
            )
            prob = prob[:, -1, :]
            # prob: (bs, n_vocab)

            # if self.decode_algo == 'greedy':
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            
            # elif self.decode_algo == 'multinomial':
            #     next_word = torch.multinomial(prob, 1)
            #     ys = torch.cat([ys, next_word], dim=1)
            #     next_word = torch.squeeze(next_word)

            end_condition = (next_word.to('cpu') == self.TRG.vocab.stoi['<eos>'])
            break_condition = (break_condition | end_condition)
            
            # print(i, self.max_strlen, break_condition)

            if all(break_condition):
                break
        return ys

    def sample_smiles(self, prop, src_smi, transform=True):
        """sample SMILES from properties and scaffold smiles
        
        Args:
            prop (List[n, self.cond_dim]): target properties
            src_smi (str): scaffold SMILES
            z (torch.FloatTensor): the latent space
        
        Returns:
            smiles (List[str]): list of SMILES string generated from the model
            toklen_gen (List[int]): list of token length of the generated SMILES
            toklen (List[int]): list of token length of the latent space
        """
        # print('trg:', self.TRG.vocab.stoi)
        
        n = len(prop)
                
        # property input        
        prop = np.array(prop)
        if transform:
            prop = self.scaler.transform(prop)
        prop = torch.from_numpy(prop).float()
        # print('property:', prop)

        # input: src
        src_ids_with_se = smi_to_id(src_smi, self.TRG,
                                    add_sos=True, add_eos=True)
        src = torch.Tensor(src_ids_with_se).long().repeat(n,1)
        # print('src:', src, src.size())
        
        # decoder embedding input
        ys = self.init_y(n)
        # print('ys:', ys.size(), ys)

        toklen = self.sample_toklen(n)
        lat_toklen = self.cond_dim + len(src_ids_with_se)-2 + max(toklen) # 72=3+22+47
        # print('toklen:', toklen)
        # print('lat_toklen:', lat_toklen)
        
        # prepare latent space
        zs = self.sample_z(lat_toklen, n) # 72=3+22+47
        # print('zs:', zs.size())
        
        # prepare mask
        src_enc_mask = torch.zeros((n, 1, lat_toklen)).bool() # 72
        for i in range(n):
            src_enc_mask[i, 0, :self.cond_dim
                               +len(src_ids_with_se)-2
                               +toklen[i]] = True
        # src_enc_mask (right): prop + sca + smi
        
        src_dec_mask = torch.ones((n,1,self.cond_dim
                                      +len(src_ids_with_se))
                                  ).bool() # 27=3+22+2
        # print('mask:', src_enc_mask.size(), src_dec_mask.size())
        # src_dec_mask (left): prop + <sos>smi<eos>
        
        # move to gpu
        ys = ys.to(self.device)
        zs = zs.to(self.device)
        src = src.to(self.device)
        prop = prop.to(self.device)
        src_enc_mask = src_enc_mask.to(self.device)
        src_dec_mask = src_dec_mask.to(self.device)
        
        # sample smiles
        outs = self.decode(zs, ys, prop, src, src_enc_mask, src_dec_mask)
        outs = outs.cpu().numpy()
        # print('outs:', outs)
        
        # print('gen_ids:', gen_ids)
        
        smiles = [id_to_smi(ids, self.TRG) for ids in outs]
        toklen_gen = [len(smi) for smi in smiles]
        print('smiles:', smiles)
        print('toklen_gen:', toklen_gen)
        return smiles, toklen, toklen_gen


sampling_tools = {
    'cvaetf'     : ScaCvaetfV1Sampling,
    'scacvaetfv1': ScaCvaetfV1Sampling,
    'scacvaetfv2': ScaCvaetfV2Sampling,
}