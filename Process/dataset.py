# general modules
import os
import pandas as pd
from multiprocessing import Pool

# ml modules
import moses
# from torchtext import data
from torchtext.legacy import data

# other packages
from utils.compute_property import property_prediction
from configuration.config_default import MAX_STRLEN


def get_dataset(data_name, data_type='train') -> pd.DataFrame:
    """
    Get the benchmarking dataset represented as strings
    Returns a DataFrame with columns ['src_no', 'trg_no', 'src', 'trg']

    Arguments:
        data_name: 'moses' or 'guacamol'
        data_type: 'train' or 'test'
    """
    if data_name == 'moses':
        dataset = moses.get_dataset(data_type)
    elif data_name == 'guacamol':
        # 補
        pass
    dataset = pd.DataFrame.from_dict({
        'src_no': [i+1 for i in range(len(dataset))],
        'trg_no': [i+1 for i in range(len(dataset))],
        'src': dataset,
        'trg': dataset
    })
    return dataset


def get_condition(dataset, condition_list,
                  condition_path=None, n_jobs=1) -> pd.DataFrame:
    """
    Compute the properties of the source and target in dataset
    Return a DataFrame of given properties

    Arguments:
        dataset:
        condition_list:
        condition_path: the conditions to download
        n_jobs: number of processes
    """
    if condition_path is not None:
        try:
            condition_library = pd.read_csv(condition_path)
        except:
            print("File not Found:", condition_path)
            exit(1)
        # 補
        return condition_library

    pool = Pool(n_jobs)
    condition_dict = {}
    for prop in condition_list:
        predictor = property_prediction[prop]
        condition_dict[f"src_{prop}"] = list(pool.map(predictor, dataset['src']))
        condition_dict[f"trg_{prop}"] = list(pool.map(predictor, dataset['trg']))

    return pd.DataFrame.from_dict(condition_dict)


def mask_invalid_len_data(dataset, num_conds, lang_format, max_strlen):
    if lang_format == 'SMILES':
        mask = (dataset['src'].str.len() + num_conds < max_strlen) \
             & (dataset['trg'].str.len() + num_conds < max_strlen)
    elif lang_format == 'SELFIES':
        mask = (dataset['src'].str.count('][') + num_conds < max_strlen) \
            & (dataset['trg'].str.count('][') + num_conds < max_strlen)
    return dataset.loc[mask]


def data_augmentation(dataset, similarity):
    assert 0 <= similarity and similarity <= 1
    # 補
    return dataset


def get_dataset_dataframe(data_name, data_type, condition_list, condition_path=None,
                          similarity=1, lang_format='SMILES', n_jobs=1, n_samples=None, 
                          max_strlen=MAX_STRLEN) -> pd.DataFrame:
    dataset = get_dataset(data_name, data_type)
    dataset = mask_invalid_len_data(dataset, len(condition_list), 
                                    lang_format, max_strlen)
    # augment dataset if needed
    dataset = data_augmentation(dataset, similarity)
    if n_samples is not None:
        dataset = dataset.head(min(len(dataset), n_samples))
    
    condition = get_condition(dataset, condition_list,
                              condition_path, n_jobs)
    return pd.concat([dataset, condition], axis=1)


def to_dataset(data_path, data_fields,
               format='csv', skip_header=True):
    dataset = data.TabularDataset(data_path=data_path, format=format,
                                  fields=data_fields, skip_header=skip_header)
    return dataset


def to_tokenlist(dataset, data_path):
    toklenList = []
    for i in range(len(dataset)):
        toklenList.append(len(vars(dataset[i])['src']))
    df_toklenList = pd.DataFrame(toklenList, columns=["toklen"])
    df_toklenList.to_csv(os.path.join(data_path, "toklen_list.csv"), index=False)
