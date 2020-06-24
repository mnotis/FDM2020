# File to create pics of the snap files.
# Run from command line by passing the arguments: path_to_folder, dir_name, n_sim.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import sys

#path = 'C:/Users/"Noah Notis"/Dropbox/USRP2020_Noah/src/output/fInfL20T4n40r64/snap0000.h5'

# visualizer that doesn't save file:
def visualize_test(psi_real, psi_im, axis_num):
    psi_real2 = np.sum(psi_real**2, axis = axis_num)
    psi_im2 = np.sum(psi_im**2, axis = axis_num)
    psi2 = np.log10(psi_real2 + psi_im2) # using logarithmic scale to see color differences
    plt.set_cmap('inferno')
    plt.imshow(psi2)
    plt.colorbar()
    plt.show()
    
#takes in two 3D arrays and produces picture of the result
#saves picture in file in subdirectory
#params: two 3d arrays, the axis to sum across, 
#the figure name, and the subdirectory name.
#axis_num can be {0, 1, 2}

def visualize(psi_real, psi_im, axis_num, fig_name, dir_name):
    psi_real2 = np.sum(psi_real**2, axis = axis_num)
    psi_im2 = np.sum(psi_im**2, axis = axis_num)
    psi2 = np.log(psi_real2 + psi_im2)
    plt.set_cmap('inferno')
    plt.imshow(psi2)
    
    plt.colorbar()
    plt.savefig(dir_name + "/" + fig_name)
    plt.clf()

    #params: path to folder with hdf5 files, desired subfolder name, the simulation size, e.g. 40 (not counting initial pic)

def make_pics(path_to_folder, dir_name, n_sim): 
    dir = path_to_folder + '/' + dir_name
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(n_sim + 1):
        hf = h5py.File(path_to_folder + "/snap" + str(i).zfill(4) +".h5", 'r')
        psi_real = np.array(hf['psiRe'])
        psi_im = np.array(hf['psiIm'])
        visualize(psi_real, psi_im, 2, 'pic' + str(i).zfill(4), dir)
        
make_pics(sys.argv[1], sys.argv[2], int(sys.argv[3]))