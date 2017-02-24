'''
This is a dummy file for me to get started making an Ising model. I'll get this 2-D Ising running, then generalize.
'''

from itertools import izip
import numpy as np
from matplotlib import pyplot as plt

N = 10
d = 2
K = 0.01
J = 1
n_steps = 1000

#at its current stage, i'll only have one spin flip at a time
# I dont know if I need to make a probability matrix or just calculate probabilities on the fly.
# I think the latter because the whole matrix would be (N^2)^d which gets enormous fast.

lattice = np.ones(tuple(N for i in xrange(d)))

def get_NN(site, N=N, d=d):
    '''
    The NN of the site. Will only return those UP in index (east, south, and down) to avoid double counting.
    Accounts for PBC
    :param site:
        (d,) array of coordinates in the lattice
    :param N:
        Size of one side of the lattice
    :param d:
        dimension of the lattice
    :return:
        dxd numpy array where each row corresponds to the nearest neighbors.
    '''
    mult_sites = np.r_[ [site for i in xrange(d)]]
    adjustment = np.eye(d)
    return ((mult_sites+adjustment)%N).astype(int)


def potential(s1, s2, K=K, h=0):
    '''
    Basic Ising potential
    :param s1:
        First spin (-1 or 1)
    :param s2:
        Second spin
    :param K:
        Coupling constant
    :return:
        Energy of this particular bond
    '''
    return -1*K*s1*s2 - h/2*(s1+s2)#should this be abstracted to call the NN function?

def energy(lattice, potential):
    '''
    Calculate the energy of a lattice
    :param lattice:
        Lattice to calculate the energy on
    :param potential:
        Function defining the potential of a given site.
    :return:
        Energy of the lattice
    '''
    N = lattice.shape[0]
    d = len(lattice.shape)

    dim_slices = np.meshgrid(*(xrange(N) for i in xrange(d)), indexing = 'ij')
    all_sites = izip(*[slice.flatten() for slice in dim_slices])

    E = 0
    for site in all_sites:
        nn = get_NN(site)
        for neighbor in nn:
            E+=potential(lattice[site], lattice[tuple(neighbor)], h= 10)

    return E

def magnetization(lattice):
    return lattice.mean()

#do different initialization
np.random.seed(0)
E_0 = energy(lattice, potential)
#plt.ion()
for step in xrange(n_steps):
    site = tuple(np.random.randint(0,N, size = d))
    #consider flipping this site
    lattice[site]*=-1
    E_f = energy(lattice, potential)

    #if E_F < E_0, keep
    #if E_F > E_0, keep randomly given change of energies
    if E_f >= E_0:
        keep = np.random.uniform() < np.exp(K/J*(E_0-E_f))
    else:
        keep = True

    if keep:
        E_0 = E_f
    else:
        lattice[site]*=-1

    #plt.imshow(lattice, interpolation='none')
    #plt.title(magnetization(lattice))
    print magnetization(lattice)
    #plt.show()
    #plt.pause(0.01)

