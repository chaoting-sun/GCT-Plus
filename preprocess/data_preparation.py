import pandas as pd
import os

from sklearn.model_selection import train_test_split

import utils.file as uf
import configuration.config_default as cfgd
import preprocess.property_change_encoder as pce

SEED = 42
SPLIT_RATIO = 0.8


def get_smiles_list(file_name):
    """
    Get smiles list for building vocabulary
    :param file_name:
    :return:
    """
    pd_data = pd.read_csv(file_name, sep=",")

    print("Read %s file" % file_name)
    smiles_list = pd.unique(pd_data[['Source_Mol', 'Target_Mol']].values.ravel('K'))
    print("Number of SMILES in chemical transformations: %d" % len(smiles_list))

    return smiles_list


def split_data(input_transformations_path, LOG=None):
    """
    Split data into training, validation and test set, write to files
    :param input_transformations_path:L
    :return: dataframe
    """
    data = pd.read_csv(input_transformations_path, sep=",")
    if LOG:
        LOG.info("Read %s file" % input_transformations_path)

    train, test = train_test_split(
        data, test_size=0.1, random_state=SEED)
    train, validation = train_test_split(train, test_size=0.1, random_state=SEED)
    if LOG:
        LOG.info("Train, Validation, Test: %d, %d, %d" % (len(train), len(validation), len(test)))

    parent = uf.get_parent_dir(input_transformations_path)
    train.to_csv(os.path.join(parent, "train.csv"), index=False)
    validation.to_csv(os.path.join(parent, "validation.csv"), index=False)
    test.to_csv(os.path.join(parent, "test.csv"), index=False)

    return train, validation, test


def save_df_property_encoded(file_name, LOG=None):
    data = pd.read_csv(file_name, sep=",")
    for property_name in cfgd.PROPERTIES:
        data['Delta_{}'.format(property_name)] = data[f'Target_Mol_{property_name}'] - data[f'Source_Mol_{property_name}']

    output_file = file_name.split('.csv')[0] + '_encoded.csv'
    if LOG:
        LOG.info("Saving encoded property change to file: {}".format(output_file))
    data.to_csv(output_file, index=False)
    return output_file
