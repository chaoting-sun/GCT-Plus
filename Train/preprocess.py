import os
import argparse
import pandas as pd

from Configuration import options
from Utils import get_raw_dataset, data_augmentation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options(parser)    
    args = parser.parse_args()


    if not os.path.exists(os.path.join(args.data_path, 'train.csv')):
        dataset = get_raw_dataset(data_name=args.data_name, 
                                  data_type='train',
                                  scaler_path=args.scaler_path,
                                  condition_list=args.cond_list, 
                                  condition_path=args.condition_path,
                                  max_strlen=args.max_strlen,
                                  n_jobs=args.n_jobs)
        
        if args.similarity < 1:
            dataset = data_augmentation(dataset, args.similarity)

        dataset.to_csv(os.path.join(args.data_path, 'train.csv'), index=False)


    if not os.path.exists(os.path.join(args.data_path, 'validation.csv')):
        dataset = get_raw_dataset(data_name=args.data_name, 
                                  data_type='validation',
                                  scaler_path=args.scaler_path,
                                  condition_list=args.cond_list, 
                                  condition_path=args.condition_path,
                                  max_strlen=args.max_strlen,
                                  n_jobs=args.n_jobs)
        
        if args.similarity < 1:
            dataset = data_augmentation(dataset, args.similarity)

        dataset.to_csv(os.path.join(args.data_path, 'validation.csv'), index=False)
