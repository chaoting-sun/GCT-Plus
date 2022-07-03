import os
import moses

if __name__ == '__main__':
    inname = 'test'
    infile = '/fileserver-gamma/chaoting/ML/dataset/moses/raw/test/'

    if not os.path.exists(os.path.join(infile, f'{inname}.smi')):
        dataset = moses.get_dataset('test_scaffolds')
        with open(os.path.join(infile, f'{inname}.smi'), 'w') as writer:
            for i, smi in enumerate(dataset):
                writer.write('{} {:d}\n'.format(smi, i+1))

    inname = 'validation'
    infile = '/fileserver-gamma/chaoting/ML/dataset/moses/raw/validation/'

    if not os.path.exists(os.path.join(infile, f'{inname}.smi')):
        dataset = moses.get_dataset('test')
        with open(os.path.join(infile, f'{inname}.smi'), 'w') as writer:
            for i, smi in enumerate(dataset):
                writer.write('{} {:d}\n'.format(smi, i+1))