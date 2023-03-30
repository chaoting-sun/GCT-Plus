import torch
from functools import partial


def cvaetf_collate_fn(ins, SRC, TRG, device):
    outs = {}
    
    outs['src'] = SRC.process([e['src'] for e in ins]).to(device)
    if not SRC.batch_first:
        outs['src'] = outs['src'].T
    
    outs['trg'] = TRG.process([['trg'] for e in ins]).to(device)
    if not TRG.batch_first:
        outs['trg'] = outs['trg'].T

    for p in ['econds', 'dconds']:
        outs[p] = torch.tensor([e[p] for e in ins],
            dtype=torch.float32).to(device)
        

def scacvaetfv1_collate_fn(ins, SRC, TRG, device):
    """collate function for scascvaetf
    
    Returns:
        outs (Dict):
            src: concatenated source smiles and its scaffold
                ex: src-src_scaffold
            trg: concatenated target smiles and its scaffold
                ex: <sos>-trg-trg_scaffold-<eos>
            econds: source properties
            dconds: target properties
    """
    outs = {}
    src = trg = None
    
    src = [b['src_scaffold']+b['src'] for b in ins]
    trg = [b['trg_scaffold']+b['trg'] for b in ins]

    outs['src'] = SRC.process(src).to(device)
    if not SRC.batch_first:
        outs['src'] = outs['src'].T

    outs['trg'] = TRG.process(trg).to(device)
    if not TRG.batch_first:
        outs['trg'] = outs['trg'].T

    for prop in ['econds', 'dconds']:
        outs[prop] = torch.tensor(
            [b[prop] for b in ins],
            dtype=torch.float32).to(device)
    return outs


def scacvaetfv2_collate_fn(ins, SRC, TRG, device):
    """collate function for scascvaetf
    
    Returns:
        outs (Dict):
            src: source smiles
            src_scaffold: scaffold of source smiles
            trg: source smiles
            trg_scaffold: scaffold of target smiles
            econds: source properties
            dconds: target properties
    """
    outs = {}

    for s in ['src', 'src_scaffold']:
        outs[s] = SRC.process([e[s] for e in ins]).to(device)
        if not SRC.batch_first:
            outs[s] = outs[s].T
        # List[int]: [t1, ..., tn, <pad>, ...]
    
    for t in ['trg', 'trg_scaffold']:
        outs[t] = TRG.process([e[t] for e in ins]).to(device)
        if not TRG.batch_first:
            outs[t] = outs[t].T
        # List[int]: [<sos>, t1, ..., tn, <eos>, <pad>, ...]
        
    for p in ['econds', 'dconds']:
        outs[p] = torch.tensor([e[p] for e in ins],
            dtype=torch.float32).to(device)
    return outs


def scacvaetfv3_collate_fn(ins, SRC, TRG, device):
    outs = {}
    src = trg = None
    
    src = [b['src_scaffold']+['<sep>']+b['src'] for b in ins]
    trg = [b['trg_scaffold']+['<sep>']+b['trg'] for b in ins]
    outs['src'] = SRC.process(src).to(device)
    if not SRC.batch_first:
        outs['src'] = outs['src'].T

    outs['trg'] = TRG.process(trg).to(device)
    if not TRG.batch_first:
        outs['trg'] = outs['trg'].T

    for prop in ['econds', 'dconds']:
        outs[prop] = torch.tensor(
            [b[prop] for b in ins],
            dtype=torch.float32).to(device)
    return outs


collate_fn = {
    'cvaetf'     : cvaetf_collate_fn,
    'scacvaetfv1': scacvaetfv1_collate_fn,
    'scacvaetfv2': scacvaetfv2_collate_fn,
    'scacvaetfv3': scacvaetfv3_collate_fn
}


def get_collate_fn(model_type, SRC, TRG, device):
    return partial(collate_fn[model_type],
                   SRC=SRC, TRG=TRG, device=device)