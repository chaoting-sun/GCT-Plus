import matplotlib.pyplot as plt
from Utils.property import tanimoto_similarity as similarity_fcn
import os


atoms = [
    ["nothing" , ""  ],
    ["alkane"  , "C" ],
    ["oxygen"  , "O" ],
    ["chlorine", "Cl"],
    ["nitrogen", "N" ]
]

def compare_2_fcnal_groups(atom1, atom2):
    similarity_list = []
    for i in range(1, 46):
        alkane = 'C' * i + atom1
        alcohol = 'C' * i + atom2
        sim = similarity_fcn(alkane, alcohol)
        similarity_list.append(sim)
    return similarity_list


def hist_plot(data, figname='./1.png'):
    plt.figure()
    plt.plot(data)
    plt.savefig(figname)

save_folder = "similarity_test"
os.makedirs(save_folder, exist_ok=True)


for i in range(len(atoms)-1):
    for j in range(i, len(atoms)):
        name1 = atoms[i][0]; name2 = atoms[j][0]
        atom1 = atoms[i][1]; atom2 = atoms[j][1]
        
        similarities = compare_2_fcnal_groups(atom1, atom2)
        
        save_path = os.path.join(save_folder, f'{name1}-{name2}.png')
        hist_plot(similarities, save_path)
