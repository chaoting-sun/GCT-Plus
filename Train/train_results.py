import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def options(parser):
    # hard_constraints_opts(parser)
    parser.add_argument('-optimizer', type=str, default='Adam')
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
        '# Epoch'   : [int(i) for i in range(begin_epoch, end_epoch+1)],
        'train RCE' : train_rce,  'valid RCE' : valid_rce,
        'train KLD' : train_kld,  'valid KLD' : valid_kld,
        'train LOSS': train_loss, 'valid LOSS': valid_loss,
    })
    
    return data_loss
    
    
def plot(epoch, loss, name_list,
         title_name, lengend_name, save_path):
    plt.figure(figsize=(8, 6.5))

    plt.plot(epoch, loss[name_list[0]], label=name_list[0], linestyle='-', marker='X', color='#c42f2f', markersize=7)
    plt.plot(epoch, loss[name_list[1]], label=name_list[1], linestyle='-.', marker='D', color='#31733b', markersize=7)
    plt.plot(epoch, loss[name_list[2]], label=name_list[2], linestyle=':', marker='o', color='#4f6482', markersize=7)

    plt.legend()
    
    plt.xlabel('# Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.xticks(epoch, fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(title_name, fontsize=16)
    plt.savefig(save_path)


# def plot_results(model_folder, begin_epoch, end_epoch):
#     data_loss = get_train_results(model_folder, begin_epoch, end_epoch)
    
#     plot(epoch=data_loss['# Epoch'],
#          loss=data_loss,
#          name_list=['train RCE'],
#          title_name='Loss of Training Loss',
#          lengend_name='Training Loss',
#          save_path=os.path.join(model_folder, 'train_loss.png')
#          )

#     plot(epoch=data_loss['# Epoch'],
#          loss=data_loss,
#          name_list=['valid RCE'],
#          title_name='Loss of Validation Loss',
#          lengend_name='Validation Loss',
#          save_path=os.path.join(model_folder, 'valid_loss.png')
#          )
    

def plot_results(model_folder, begin_epoch, end_epoch):
    data_loss = get_train_results(model_folder, begin_epoch, end_epoch)
    data_loss.to_csv(os.path.join(model_folder, 'loss.csv'), index=False)

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
    
    
def plot_train_results(optimizer, model_folder, begin_epoch, end_epoch):
    data_loss = get_train_results(model_folder, begin_epoch, end_epoch)

    plt.figure(figsize=(7.5, 6))
    
    plt.plot(data_loss['# Epoch'], data_loss['train RCE'], 
             label='train RCE', linestyle='-', marker='^', color='#3c28ed', markersize=5)
    plt.plot(data_loss['# Epoch'], data_loss['train KLD'], 
             label='train KLD', linestyle='-', marker='x', color='#3c28ed', markersize=5)

    plt.plot(data_loss['# Epoch'], data_loss['valid RCE'], 
             label='valid RCE', linestyle='-', marker='^', color='#29dff0', markersize=5)
    plt.plot(data_loss['# Epoch'], data_loss['valid KLD'],
             label='valid KLD', linestyle='-.', marker='x', color='#29dff0', markersize=5)

    plt.legend()
    
    plt.title(f'Training/Validation Loss ({optimizer})', fontsize=19)
    plt.xlabel('# Epoch', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    plt.xticks(fontsize=16.5)
    plt.yticks(fontsize=16.5)
    plt.legend(fontsize=16)
    
    plt.savefig(os.path.join(model_folder, 'train_loss.png'))


def plot_ctf_train_results(model_folder, begin_epoch, end_epoch):
    def data_results(data_name):
        print('dataset:', data_name)
        rce = []
        for i in range(begin_epoch, end_epoch+1):
            data = pd.read_csv(os.path.join(model_folder, f"{data_name}_{i}.csv"))
            rce.append(data['RCE'].mean())
            print(f'{i}\trce: {rce[-1]:.5f}')
        return rce

    train_rce = data_results('train')
    valid_rce = data_results('valid')

    data_loss = pd.DataFrame({
        '# Epoch'   : [int(i) for i in range(begin_epoch, end_epoch+1)],
        'train RCE' : train_rce,  'valid RCE' : valid_rce,
    })

    plt.figure(figsize=(7.5, 6))
    
    plt.plot(data_loss['# Epoch'], data_loss['train RCE'], 
             label='train RCE', linestyle='-', marker='x', color='#3c28ed', markersize=5)
    plt.plot(data_loss['# Epoch'], data_loss['valid RCE'], 
             label='valid RCE', linestyle='-', marker='x', color='#29dff0', markersize=5)
    plt.legend()
    
    plt.title(f'Training/Validation Loss', fontsize=19)
    plt.xlabel('# Epoch', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    plt.xticks(fontsize=16.5)
    plt.yticks(fontsize=16.5)
    plt.legend(fontsize=16)
    
    plt.savefig(os.path.join(model_folder, 'train_loss.png'))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options(parser)
    args = parser.parse_args()
    
    # plot_train_results(args.optimizer, args.model_folder, args.begin_epoch, args.end_epoch)
    plot_results(args.model_folder, args.begin_epoch, args.end_epoch)

    # plot_ctf_train_results(args.model_folder, args.begin_epoch, args.end_epoch)