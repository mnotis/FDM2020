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
   
# takes a two dimensional density array and centers it on the highest point
def rho_roll(rho):
    rho_len = rho.shape[0]
    # finding the maximum value's index
    max_ind = np.where(rho == np.amax(rho))
    x_max = max_ind[0][0]
    y_max = max_ind[1][0]
    #z_max = max_ind[2][0]
    signed_x_dist_to_cent = int(rho_len / 2) - x_max
    signed_y_dist_to_cent = int(rho_len / 2) - y_max
    #signed_z_dist_to_cent = int(rho_len / 2) - z_max
    #moving the core to the center of the cube
    rho_n = np.roll(rho, signed_x_dist_to_cent, axis=0)
    rho_n2 = np.roll(rho_n, signed_y_dist_to_cent, axis=1)
    #rho_n3 = np.roll(rho_n2, signed_z_dist_to_cent, axis=2)
    rho = rho_n2
    return rho

#takes in two 3D arrays and produces picture of the result
#saves picture in file in subdirectory
#params: two 3d arrays, the axis to sum across, 
#the figure name, and the subdirectory name.
#axis_num can be {0, 1, 2}

def visualize(psi_real, psi_im, axis_num, fig_name, dir_name):
    psi_real2 = np.sum(psi_real**2, axis = axis_num)
    psi_im2 = np.sum(psi_im**2, axis = axis_num)
    psi2 = np.log(psi_real2 + psi_im2)
    psi2 = rho_roll(psi2)
    plt.set_cmap('inferno')
    plt.imshow(psi2)
    
    plt.colorbar()
    plt.savefig(dir_name + "/" + fig_name)
    plt.clf()
    
#params: path to folder with hdf5 files, desired subfolder name, the simulation size, e.g. 40 (not counting initial pic)

def make_pics(path_to_folder, dir_name, n_sim): 
    dir = dir_name
    n_sim = int(n_sim)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(0, n_sim + 1):
        hf = h5py.File(path_to_folder + "/snap" + str(i * 10).zfill(4) +".h5", 'r')
        psi_real = np.array(hf['psiRe'])
        psi_im = np.array(hf['psiIm'])
        visualize(psi_real, psi_im, 2, 'pic' + str(i * 10).zfill(4), dir)
        
path_to_folder = '/tigress/mnotis/f1L20T4n400r256'
dir_name = 'pics/f1L20T4n400r256'
n_sim = 40
print(path_to_folder)
print(dir_name)
print(n_sim)
make_pics(path_to_folder, dir_name, n_sim)
print("Task completed successfully!")
