# """
# Preprocess
# - encode property change
# - build vocabulary
# - split data into train, validation and test
# """
import os
import argparse
import pickle

# import Preprocess.vocabulary as mv
# import Preprocess.data_preparation as pdp
# import configuration.config_default as cfgd
# import utils.log as ul
# import utils.file as uf
# import Preprocess.property_change_encoder as pce

import pandas as pd
import Process.data_preparation as pdp

import moses


# global LOG
# LOG = ul.get_logger("preprocess", "experiments/preprocess.log")


def parse_args():
    """Parses arguments from cmd"""
    parser = argparse.ArgumentParser()
    # parser.add_argument("-input_data_path", "-i", type=str, required=True)
    parser.add_argument('-load_weights', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-checkpoint', type=int, default=0)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-verbose', type=bool, default=True)
    parser.add_argument('-device', type=bool, default=True)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-data_folder', type=str, default='.')
    parser.add_argument('-cond_list', nargs='+', default=['logP', 'tPSA', 'QED'], help="Conditions")
    
    return parser.parse_args()


# # def ii(file_path, save_name, tokenizer, fields):
# #     df_file = pd.read_csv(file_path)
# #     SEQ = data.Field(tokenize=tokenizer, batch_first=True)
# #     SEQ.build_vocab(df_file, max_size=150)

train_prop_path = "data/moses/train_prop.csv"
test_prop_path = "data/moses/test_prop.csv"

def main():
    opt = parse_args()

    if opt.verbose:
        print("Create fields of source and target SMILES.")
    SRC, TRG = pdp.create_seq_fields(opt.lang_format, None)

    print(type(SRC))

    if opt.verbose:
        print("Get the dataset and its properties.")
    opt.source = moses.get_dataset('train')[:256]
    opt.target = opt.source
    opt.conds = pd.read_csv(train_prop_path)

    data_iter = pdp.create_dataset(opt, SRC, TRG, tr_te="train")

    from Process.batch import rebatch

    print("test1")
    train_iter = (rebatch(opt.src_pad, b) for b in data_iter)
    print("test2")

    print(type(train_iter))

    for i, batch in enumerate(train_iter):
        print("source:")
        print(batch.src)
        print("target:")
        print(batch.trg)
        print("trg_y:")
        print(batch.trg_y)
        break


if __name__ == "__main__":
    main()


    args = parse_args()

#     # encoded_file = pdp.save_df_property_encoded(args.input_data_path, LOG)

#     # LOG.info("Building vocabulary")

#     # tokenizer = mv.SMILESTokenizer()
#     # smiles_list = pdp.get_smiles_list(args.input_data_path)
#     # vocabulary = mv.create_vocabulary(smiles_list, tokenizer=tokenizer)
#     # tokens = vocabulary.tokens()
#     # LOG.info("Vocabulary contains %d tokens: %s", len(tokens), tokens)

#     # # Save vocabulary to file
#     # parent_path = uf.get_parent_dir(args.input_data_path)
#     # output_file = os.path.join(parent_path, 'vocab.pkl')
#     # with open(output_file, 'wb') as pickled_file:
#     #     pickle.dump(vocabulary, pickled_file)
#     # LOG.info("Save vocabulary to file: {}".format(output_file))

#     # # Split data into train, validation, test
#     # train, validation, test = pdp.split_data(encoded_file, LOG)


#     # data_path = "."
#     # train_path = "train.csv"
#     # valid_path = "validation.csv"
#     # test_path = "test.csv"

#     # SRC = data.Field(tokenize=tokenizer, batch_first=True)
#     # SRC.build_vocab(df_file, max_size=150)

#     # TRG = data.Field(tokenize=tokenizer, batch_first=True)
#     # TRG.build_vocab(df_file, max_size=150)

#     # tokenizer = moltokenize()

#     # fields = [("SMILES", SRC)]
    

#     # train_data, valid_data, test_data = data.TabularDataset.splits(
#     #     path = data_path,
#     #     train = train_path,
#     #     validation = valid_path,
#     #     test = test_path,
#     #     format = "csv",
#     #     fields = fields,
#     #     skip_header = True
#     # )