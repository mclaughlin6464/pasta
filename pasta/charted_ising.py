'''
This is a dummy file for me to get started making an Ising model. I'll get this 2-D Ising running, then generalize.
'''

import argparse
from itertools import izip
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def run_ising(N, d, K, J,h, n_steps, plot = False):
    '''

    :param N:
    :param d:
    :param K:
    :param J:
    :param h:
    :param n_steps:
    :param plot:
    :return:
    '''

    if plot:
        try:
            assert d <= 2
        except AssertionError:
            raise AssertionError("Can only plot in one or two dimensions.")

    #TODO wrap these better
    assert N >0 and N < 1000
    assert d > 0
    assert n_steps > 0

    np.random.seed(0)

    size = tuple(N for i in xrange(d))
    lattice = np.ones(size)
    #make a random initial state
    lattice-= np.random.randint(0,2, size =size)*2

    # do different initialization
    E_0 = energy(lattice, potential, K, h)
    if plot:
        plt.ion()
    for step in xrange(n_steps):
        if step%1000 == 0:
            print step
        site = tuple(np.random.randint(0, N, size=d))
        # consider flipping this site
        lattice[site] *= -1
        E_f = energy(lattice, potential, K, h)

        # if E_F < E_0, keep
        # if E_F > E_0, keep randomly given change of energies
        if E_f >= E_0:
            keep = np.random.uniform() < np.exp(K / J * (E_0 - E_f))
        else:
            keep = True

        if keep:
            E_0 = E_f
        else:
            lattice[site] *= -1

        # fig = plt.figure()
        if plot and step % 100 == 0:
            if d == 1:
                plt.imshow(lattice.reshape((1, -1)),interpolation='none')
            else:
                plt.imshow(lattice, interpolation='none')
            plt.title(correlation(lattice, N/2))
            plt.pause(0.01)
            plt.clf()

    return np.array([correlation(lattice, r) for r in xrange(1, N/2+1)])

def get_NN(site, N, d, r= 1):
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
    adjustment = np.eye(d)*r
    return ((mult_sites+adjustment)%N).astype(int)


def potential(s1, s2, K, h):
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

def energy(lattice, potential, K, h = 0):
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
        nn = get_NN(site, N, d)
        for neighbor in nn:
            E+=potential(lattice[site], lattice[tuple(neighbor)],K = K, h = h)

    return E

def magnetization(lattice):
    return lattice.mean()

def correlation(lattice, r):
    '''
    The average spin correlation at distance r.
    :param lattice:
        The lattice to calculate the statistic on.
    :param r:
        Distance to measure correlation
    :return:
    '''
    N = lattice.shape[0]
    d = len(lattice.shape)

    dim_slices = np.meshgrid(*(xrange(N) for i in xrange(d)), indexing='ij')
    all_sites = izip(*[slice.flatten() for slice in dim_slices])

    xi = 0
    for site in all_sites:
        nn = get_NN(site, N, d, r)
        for neighbor in nn:
            xi += lattice[site]*lattice[tuple(neighbor)]

    return xi/((N**d)*d)

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an ising model')
    parser.add_argument('N', type = int, help = 'Length of one side of the cube.')
    parser.add_argument('d', type = int, help = 'Number of dimensions of the cube.')
    #parser.add_argument('K', type = float, help ='Bond coupling strength.')

    parser.add_argument('J', type = float, default = 1.0, nargs = '?',\
                        help = 'Energy of bond strength. Optional, default is 1.')
    parser.add_argument('h', type = float, default=0.0, nargs = '?',\
                        help = 'Magnetic field strength. Optional, default is 0.')
    parser.add_argument('n_steps', type = int, default = 1000, nargs = '?',\
                        help = 'Number of steps to simulate. Default is 1e5')
    parser.add_argument('--plot', action = 'store_true',\
                        help = 'Whether or not to plot results. Only allowed with d = 1 or 2.')

    args = parser.parse_args()
    spins = []
    Ks = [ 0.5,0.6,0.65, 0.7,0.8, 0.9]
    for K in Ks:
        print K
        spins.append(run_ising(K = K, **vars(args)))

    for K, spin in izip(Ks, spins):
        plt.plot(spin, label = K )
    plt.legend(loc = 'best')
    plt.ylim([-0.1, 1.1])
    plt.show()