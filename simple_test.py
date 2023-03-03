import sys
import pandas as pd
from Utils.properties import predict_props


def predict_several_props(smiles_list):
    for smiles in smiles_list:
        print(smiles, predict_props(smiles))


def data_analysis(data_type):
    data_path = f'/fileserver-gamma/chaoting/ML/dataset/moses/aug/data_sim0.50_tol0.20/{data_type}.csv'
    df = pd.read_csv(data_path)
    
    df_same = df[df.src_no == df.trg_no]
    df_not_same = df[df.src_no != df.trg_no]

    print(f'# same: {len(df_same)}\t# not same: {len(df_not_same)}')

    df_not_same_but_same_props = df_not_same[(df_not_same.src_logP == df_not_same.trg_logP) &
                                             (df_not_same.src_tPSA == df_not_same.trg_tPSA) &
                                             (df_not_same.src_QED == df_not_same.trg_QED)
                                             ]
    
    
    print(f'#no same - same props: {len(df_not_same_but_same_props)}, '
          f'not same props {len(df_not_same) - len(df_not_same_but_same_props)}')


if __name__ == '__main__':
    # assert len(sys.argv) >= 2
    
    # smiles_list = sys.argv[1:]
    # predict_several_props(smiles_list)
    
    data_analysis('train')
    data_analysis('validation')