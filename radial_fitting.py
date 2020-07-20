#Python program to be run from command line that takes the name of a snapshot and saves a picture
# of the radial profile along with a best fit analytic profile.

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
from scipy.optimize import minimize


# function takes in a path and outputs a 3d array of the density profile.

def file_to_density(path_name):
    hf = h5py.File(path_name, 'r')
    psi_real = np.array(hf['psiRe'])
    psi_im = np.array(hf['psiIm'])
    psi_real2 = psi_real**2
    psi_im2 = psi_im**2
    rho = (psi_real2 + psi_im2)
    return rho

#function takes in a cubic np density array and rolls it so tha the highest point is in the center.
#only to be used for the steady state images

def rho_roll(rho):
    rho_len = rho.shape[0]
    # finding the maximum value's index
    max_ind = np.where(rho == np.amax(rho))
    x_max = max_ind[0][0]
    y_max = max_ind[1][0]
    z_max = max_ind[2][0]
    signed_x_dist_to_cent = int(rho_len / 2) - x_max
    signed_y_dist_to_cent = int(rho_len / 2) - y_max
    signed_z_dist_to_cent = int(rho_len / 2) - z_max
    #moving the core to the center of the cube
    rho_n = np.roll(rho, signed_x_dist_to_cent, axis=0)
    rho_n2 = np.roll(rho_n, signed_y_dist_to_cent, axis=1)
    rho_n3 = np.roll(rho_n2, signed_z_dist_to_cent, axis=2)
    rho = rho_n3
    return rho

# function that calculates distances between two 3d points:

def get_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2]- b[2])**2)

#function that takes in a 3d array with highest point at the center and calculates
#the density as a function of radius fromt the center of the core. returns as a dictionary with
#distances as keys and a list of the densities as the value.

def radial_density(rho):
    rho_len = rho.shape[0]
    center = int(rho_len / 2)
    radii_den = {}
    for i in range(0, rho_len):
        for j in range(0, rho_len):
            for k in range(0, rho_len):
                dist = get_distance([center, center, center], [i, j, k])
                dens = rho[i][j][k]
                if dist not in radii_den:
                    radii_den[dist] = [dens]
                else:
                    radii_den[dist].append(dens)
    return radii_den

# testing
#a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#b = np.array([[[1, 4], [3, 4]], [[5, 6], [7, 8]]])
#dict = radial_density(a)
#print(len(dict))
#dist = np.fromiter(dict.keys(), dtype=float)
#print(dist)

#function that plots radial density. Takes in a dictionary with densities as keys
# and the densities as values and outputs two np arrays, one with the distances 
# and one with the ---average--- density at that radial distance.

def rad_avg_den(dict, scale_factor):
    dict_len = len(dict)
    dist = np.fromiter(dict.keys(), dtype=float)
    avgs = np.zeros(dict_len)
    for x in range(0, dict_len):
        dens = np.array(dict[dist[x]])
        avg = np.mean(dens)
        avgs[x] = avg
    return dist * scale_factor, avgs

# testing
#res = rad_avg_den(dict)
#print(res[0])
#print(res[1])

# from path to distance and average density arrays
def file_to_rad_avg_den(path):
    rho = file_to_density(path)
    rho = rho_roll(rho)
    scale_factor = 20/ rho.shape[0]
    rad_den = radial_density(rho)
    return rad_avg_den(rad_den, scale_factor)

#file to picture
def file_to_pic(path_to_dir, fig_num, f_str):
    res = file_to_rad_avg_den(path_to_dir + '/snap' + fig_num  + '.h5')
    plt.scatter(res[0], res[1])
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Density (Msun / (kpc)^3)')
    plt.title('Radial Distance vs. Density: f = ' + f_str)
    plt.ylim(10**4, 10**(10.5))
    plt.xlim(10**(-1), 10**(2.2))
    plt.savefig(path_to_dir + "/" + f_str + 'pic' + fig_num)
    plt.clf()
    
#cuts off a distance array such that it only has elements < a certain length
def dist_cutoff(dist, dist_lim):
    distance = []
    for x in dist:
        if (x < dist_lim):
            distance.append(x)
    return np.asarray(distance)

#cuts off a density array at a given number of elements
#not in place
def dens_cutoff(dens, n_elements):
    densities = np.zeros(n_elements)
    for i in range(0, n_elements):
        densities[i] = dens[i]
    return densities

#analytic profile of the core
def an_prof(r, rc):
    rho = (1.9*10**7 / (rc**4))
    term = (1 + 0.091*(r/rc)**2)**-8
    return rho * term

#X2 function
def X2_fun(rc, dist, dens, dist_lim):
    r = dist_cutoff(dist, dist_lim)
    densities = dens_cutoff(dens, len(r))
    X2 = (((an_prof(r, rc)) - densities)**2)
    indices = [0, len(X2)-1]
    return X2[indices].sum()

#radial-density fitting function
#params are two nparrays
#analytic radial profile function
def fit_profile(dist, dens, dist_lim):
    dist1 = np.sort(dist)
    dens1 = (-1)*(np.sort(dens*(-1)))
    params = (dist1, dens1, dist_lim)
    x0 = np.asarray(3)
    result = minimize(X2_fun, x0, args = params)
    return result

#Takes in a file path and write a picture.
def file_to_pic_curve_fit(path, file, dist_lim, f):
    arrs = file_to_rad_avg_den(path + '/' + file)
    dist = arrs[0]
    dens = arrs[1]
    dist1 = np.sort(dist)
    dens1 = (-1)*(np.sort(dens*(-1)))
    res = fit_profile(dist1, dens1, float(dist_lim))
    rc = res.x[0]
    plt.scatter(dist1, dens1)
    plt.plot(dist1, an_prof(dist1, rc))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radial Distance (kpc)')
    plt.ylabel('Density (Msun / (kpc)^3)')
    plt.title('Radial Distance vs. Density (f = ' + f + ')')
    plt.ylim(10**4, 10**(12.5))
    plt.xlim(10**(-2), 10**(1.4))
    textstr = 'rc = ' + "%.2f" % rc
    plt.legend(['Analytic Profile, rc = ' + "%.2f" % rc,'Simulation'])
    plt.savefig('pic' + file + f + '.png')
    
# writes radial h5 files for all files in folder:
def radial_h5_folder(path, n_snap):
    dir = '/home/mnotis/FDM2020' + '/rad_files/f1L20T4n40r512'
    n_snap = n_snap
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i in range(16, int(n_snap)+1):
        result = file_to_rad_avg_den(path + '/snap' + str(i).zfill(4) + '.h5')
        print(i)
        dist = np.sort(result[0])
        dens = (-1)*(np.sort(result[1]*(-1)))
        dist_lim = 0.65
        params = (dist, dens, dist_lim)
        x0 = np.asarray(2)
        res = minimize(X2_fun, x0, args = params)
        rc = res.x[0]
        # saving as file
        hf = h5py.File(dir + '/radial' + str(i).zfill(4) + '.h5', 'w')
        hf.create_dataset('distances', data=dist)
        hf.create_dataset('densities', data=dens)
        hf.create_dataset('rc', data=rc)
        hf.close()
path = '/tigress/mnotis/f1L20T4n40r512'
#file_to_pic_curve_fit(path, 'snap0400.h5', 0.65, 'Inf')       
radial_h5_folder(path, 18)
print('Task completed successfully!')
