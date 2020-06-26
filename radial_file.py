#python script that takes in a folder of h5 files and makes a folder of distance/density graph
import numpy as np
import os
import h5py
import sys
import matplotlib.pyplot as plt

def file_to_pic_curve_fit(path, n_sim):
    dir = path + '/rad_files'
    dir_pics = path + '/rad_graphs'
    if not os.path.exists(dir_pics):
        os.mkdir(dir_pics)
    for i in range(0, int(n_sim) + 1):
        hf = h5py.File(dir + '/radial' + str(i).zfill(4) + '.h5', 'r')
        dist = np.array(hf['distances'])
        dens = np.array(hf['densities'])
        rc = np.array(hf['rc'])
        plt.scatter(dist, dens)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Radial Distance (kpc)')
        plt.ylabel('Density (Msun / (kpc)^3)')
        plt.title('Radial Distance vs. Density')
        plt.ylim(10**4, 10**(12.5))
        plt.xlim(10**(-2), 10**(1.4))
        plt.legend(['Simulation, rc = ' + "%.2f" % rc])
        plt.savefig(dir_pics + '/graph' + str(i).zfill(4) + '.png')
        plt.clf()

path = sys.argv[1]
n_sim = sys.argv[2]
file_to_pic_curve_fit(path, n_sim)
