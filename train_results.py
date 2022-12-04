import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# from Configuration.config import hard_constraints_opts


def options(parser):
    # hard_constraints_opts(parser)
    parser.add_argument('-begin_epoch', type=int)
    parser.add_argument('-end_epoch', type=int)
    parser.add_argument('-model_folder', type=str)
    
    
def get_train_results(model_path, begin_epoch, end_epoch):
    print('model path:', model_path)
    
    def data_results(data_name):
        print(f'{data_name}')
        rce, kld, loss = [], [], []
        for i in range(begin_epoch, end_epoch+1):
            data = pd.read_csv(os.path.join(model_path, 
                               f"{data_name}_{i}.csv"))
            rce.append(data['RCE'].mean())
            kld.append(data['KLD'].mean())
            loss.append(data['LOSS'].mean())
            print(f'{i}\trce: {rce[-1]:.2f}\tkld: {kld[-1]:.2f}\tloss: {loss[-1]:.2f}')
        return rce, kld, loss

    train_rce, train_kld, train_loss = data_results('train')
    valid_rce, valid_kld, valid_loss = data_results('valid')

    data_loss = pd.DataFrame({
        '# Epoch': [i for i in range(begin_epoch, end_epoch+1)],
        'train RCE' : train_rce,  'valid RCE' : valid_rce,
        'train KLD' : train_kld,  'valid KLD' : valid_kld,
        'train LOSS': train_loss, 'valid LOSS': valid_loss,
    })
    
    return data_loss
    
    
def plot(epoch, loss, name_list,
         title_name, lengend_name, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(x=epoch, y=loss[name_list[0]], label=name_list[0], color='green')
    plt.scatter(x=epoch, y=loss[name_list[1]], label=name_list[1], color='steelblue')
    plt.scatter(x=epoch, y=loss[name_list[2]], label=name_list[2], color='purple')

    plt.legend(title=title_name)
    
    plt.xlabel('# Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title(title_name, fontsize=16)
    plt.savefig(save_path)


def plot_results(model_folder, begin_epoch, end_epoch):
    data_loss = get_train_results(model_folder, begin_epoch, end_epoch)
    
    plot(epoch=data_loss['# Epoch'],
         loss=data_loss,
         name_list=['train RCE', 'train KLD', 'train LOSS'],
         title_name='Loss of Training Loss',
         lengend_name='Training Loss',
         save_path=os.path.join(model_folder, 'train_loss.png')
         )

    plot(epoch=data_loss['# Epoch'],
         loss=data_loss,
         name_list=['valid RCE', 'valid KLD', 'valid LOSS'],
         title_name='Loss of Validation Loss',
         lengend_name='Validation Loss',
         save_path=os.path.join(model_folder, 'valid_loss.png')
         )
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()

    plot_results(args.model_folder, args.begin_epoch, args.end_epoch)