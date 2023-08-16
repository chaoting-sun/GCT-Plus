import os
import re
import torch
import dill as pickle
from torchtext import data


class moltokenize:
    # change from atomwise_tokenizer in SmilesPE.pretokenizer

    def __init__(self, add_sep=False):
        if add_sep:
            self.tokenizer = self._tokenizer_with_sep
        else:
            self.tokenizer = self._tokenizer
        generaral_pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(generaral_pattern)

    def _tokenize(self, sentence):
        return [token for token in self.regex.findall(sentence)]

    def _tokenizer(self, sentence):
        return [tok for tok in self._tokenize(sentence) if tok != " "]

    def _tokenizer_with_sep(self, sentence):
        pattern = re.compile(r'(<sep>)')
        res = pattern.split(sentence)
        if len(res) == 1:
            return self._tokenizer(sentence) # no <sep>
        elif len(res) == 3:
            return self._tokenize(res[0]) + ['<sep>'] + self._tokenize(res[2])
        else:
            return []

    @staticmethod
    def untokenizer(tokens, sos_idx, eos_idx, itos):
        smi = ""
        for token in tokens:
            if token == eos_idx:
                break
            elif token != sos_idx:
                smi += itos[token]
        return smi


def smiles_fields(smiles_field_path=None, add_sep=False):
    t_src = moltokenize(add_sep)
    t_trg = moltokenize(add_sep)

    SRC = data.Field(tokenize=t_src.tokenizer, batch_first=True)
    TRG = data.Field(tokenize=t_trg.tokenizer, batch_first=True,
                     init_token='<sos>', eos_token='<eos>')
    
    if smiles_field_path is not None:
        try:
            SRC = pickle.load(open(os.path.join(smiles_field_path, 'SRC.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(smiles_field_path, 'TRG.pkl'), 'rb'))
        except:
            print(">>> Files SRC.pkl/TRG.pkl not in:" + smiles_field_path)
            exit(1)

    return (SRC, TRG)


# torch.float32 is equal to torch.float
def condition_fields(conditions):
    return [data.Field(use_vocab=False, sequential=False,
            batch_first=True, dtype=torch.float32) for _ in conditions]


def get_inference_fields(conds, smiles_fields_path):
    SRC, TRG = smiles_fields(smiles_fields_path)
    PROP = condition_fields(conds)
    fsmiles = [('src', SRC)]
    fconds = [(f'src_{conds[i]}', PROP[i])
               for i in range(len(conds))] + \
              [(f'trg_{conds[i]}', PROP[i])
               for i in range(len(conds))]
    return fsmiles + fconds, SRC, TRG


def save_fields(src_fields, trg_fields, field_path):
    """ save SRC/TRG fields """
    os.makedirs(field_path, exist_ok=True)
    src_field_path = os.path.join(field_path, 'SRC.pkl')
    trg_field_path = os.path.join(field_path, 'TRG.pkl')

    if os.path.exists(src_field_path):
        exit(f'File already existed: {src_field_path}')
    if os.path.exists(trg_field_path):
        exit(f'File already existed: {trg_field_path}')
    
    pickle.dump(src_fields, open(src_field_path, 'wb'))
    pickle.dump(trg_fields, open(trg_field_path, 'wb'))


# def smiles_field(properties, field_path=None, suffix=None):
def smiles_field(field_path=None, add_sep=False):
    t_src = moltokenize(add_sep)
    t_trg = moltokenize(add_sep)
    
    SRC = data.Field(
        tokenize=t_src.tokenizer,
        batch_first=True,
        unk_token='<unk>'
    )
    
    TRG = data.Field(
        tokenize=t_trg.tokenizer,
        batch_first=True,
        init_token='<sos>',
        eos_token='<eos>',
        unk_token='<unk>'
    )
    

    if field_path is not None:
        suffix = '_sep' if add_sep else ''
        try:
            SRC = pickle.load(open(os.path.join(field_path, f'SRC{suffix}.pkl'), 'rb'))
            TRG = pickle.load(open(os.path.join(field_path, f'TRG{suffix}.pkl'), 'rb'))
        except:
            print(">>> Files SRC.pkl/TRG.pkl not in: " + os.path.join(field_path, f'SRC{suffix}.pkl'))
            exit(1)

    return (SRC, TRG)   


# torch.float32 is equal to torch.float
def property_field(property_list):
    PROP = [data.Field(use_vocab=False, sequential=False, batch_first=True,
            dtype=torch.float32) for _ in property_list]
    return PROP


def get_train_field(property_list, field_path=None):
    SRC, TRG = smiles_field(field_path, add_sep=False)
    PROP = property_field(property_list)

    src_fields = [('src', SRC)]
    trg_fields = [('trg', TRG)]
    
    prop_dict = { property_list[i]: PROP[i]
                  for i in range(len(property_list)) }
    for prop in ('logP', 'tPSA', 'QED', 'SAS'):
        if prop in property_list:
            src_fields.append((f'src_{prop}', prop_dict[prop]))
            trg_fields.append((f'trg_{prop}', prop_dict[prop]))
        else:
            src_fields.append((f'src_{prop}', None))
            trg_fields.append((f'trg_{prop}', None))
    return 


def get_iter_field(property_list, field_path=None):
    
    SRC, TRG = smiles_field(property_list, field_path)
    
    PROP = property_field(property_list)
    
    src_field = [('src', SRC)]
    src_field.extend([(f'src_{property_list[i]}', PROP[i])
                      for i in range(len(property_list))])
    trg_field = [('trg', TRG)]
    trg_field.extend([(f'trg_{property_list[i]}', PROP[i])
                      for i in range(len(property_list))])
    
    return src_field + trg_field, SRC, TRG


def untokenize(tokens, vocab):
    smi = ""
    for token in tokens:
        if token == vocab.stoi['<eos>']:
            break
        elif token != vocab.stoi['<sos>']:
            smi += vocab.itos[token]
    return smi


# def id_to_smi(ids, TRG):
#     """Convert ids into smiles based on TRG.
    
#     Args:
#         ids (List[int]): a list of integers representing a SMILES
#         TRG (torchtext.data.Field): field for target

#     Returns:
#         smi (str): a string representing a molecule                 
#     """
#     smi = ''
#     for i in ids:
#         if i == TRG.vocab.stoi['<eos>']:
#             break
#         if i != TRG.vocab.stoi['<sos>']:
#             smi += TRG.vocab.itos[i]
#     return smi


# def smi_to_id(smi, TRG, add_sos=False, add_eos=False):
#     """Convert smiles into ids based on TRG.
    
#     Args:
#         smi (str): a string representing a molecule
#         TRG (torchtext.data.Field): field for target

#     Returns:
#         ids (List[int]): a list of integers representing a SMILES
#     """
#     token = TRG.tokenize(smi)
#     ids = []
#     if add_sos:
#         ids.append(TRG.vocab.stoi['<sos>'])
#     ids.extend([TRG.vocab.stoi[t] for t in token])
#     if add_eos:
#         ids.append(TRG.vocab.stoi['<eos>'])    
#     return ids

