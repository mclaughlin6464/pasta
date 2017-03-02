'''
I'm going to modify my model for the simple ising model to the charged ising model, from Hasnaoui et al.
'''

import argparse
from time import time
from itertools import izip, product
import numpy as np
from scipy.special import erfc
from matplotlib import pyplot as plt
#import seaborn as sns
#sns.set()

d = 2 #fixed in this model
v = {(1,1): 5.167, (-1, -1): 5.167, (1, -1): -5.5, (-1, 1): -5.5 } #MeV
a = 1.842e-15 #lattice spaccing in m
a_s = 1.0 #dimensionaless smoothing length
L_MAX = 10 #TODO try other values

dim_pot_dict = {} #store old results for the dimensionless potential

site_idxs = {}

def run_ising(N, B,xb, xp, n_steps, plot = False):
    '''
    Run the simulation at temperature B for n_steps
    :param N: Int, length of one side
    :param B: Float, inverse of temperature times boltzman constant
    :param A: Int, The number of baryons in the simulation
    :param xp: float, the proton fraction
    :param n_steps: int, Number of steps
    :param plot: bool, whether or not to show a plot
    :return: None
    '''

    #TODO output writing
    #TODO autocorrelation sampling
    #TODO high temp burn-in

    if plot:
        try:
            assert d <= 2
        except AssertionError:
            raise AssertionError("Can only plot in one or two dimensions.")

    #TODO wrap these better
    assert 0< N < 100
    assert B > 0
    assert 0<= xb <=1
    assert 0 <= xp <= 1
    assert n_steps > 0

    np.random.seed(0)
    #initialize based on inputs
    size = tuple(N for i in xrange(d))
    lattice = (np.random.rand(*size) <= xb).astype(int) #occupied
    lattice*= (2*(np.random.rand(*size)<=xp) -1).astype(int)# neturon/proton fraction

    #make a list of indices
    for spin in (1,0,-1):
        site_idxs[spin] = set(tuple(idx) for idx in np.c_[(lattice == spin).nonzero()])

    site_idxs['occ'] = site_idxs[1] & site_idxs[-1] #sites occupied by nucleon

    E_0 = energy(lattice)
    if plot:
        plt.ion()
    for step in xrange(n_steps):
        if step%100 == 0:
            B*=1.1
            print step, E_0, E_0/(xb*N**3), B
            if plot:
                if d == 1:
                    im = plt.imshow(lattice.reshape((1, -1)),interpolation='none')
                else:
                    im = plt.imshow(lattice, interpolation='none')
                plt.colorbar(im)
                plt.title(r'$\beta= %e, E=%0.2f$'%(B, E_0))
                plt.pause(0.1)
                plt.clf()
        #print step
        site1, site2 = [tuple(0 for i in xrange(d)) for i in xrange(2)]
        while lattice[site1] == lattice[site2]:
            #consider flipping this site
            site1, site2 = np.random.randint(0, N, size=(2,d))
            site1, site2 = tuple(site1), tuple(site2)

        #make the appropriate switches in the sets
        site_idxs[lattice[site1]].remove(site1)
        site_idxs[lattice[site2]].remove(site2)

        site_idxs[lattice[site1]].add(site2)
        site_idxs[lattice[site2]].add(site1)

        if lattice[site1] == 0:
            #site1 is becoming occupied and site2 is becoming unoccupiated
            site_idxs['occ'].add(site1)
            site_idxs['occ'].remove(site2)
        elif lattice[site2] == 0:
            #vice versa
            site_idxs['occ'].add(site2)
            site_idxs['occ'].remove(site1)


        lattice[site1], lattice[site2] = lattice[site2], lattice[site1]
        t0 = time()
        E_f = energy(lattice)
        t1 = time()
        #print 'Total calc time: %.3f s'%(t1-t0)
        # if E_F < E_0, keep
        # if E_F > E_0, keep randomly given change of energies
        if E_f >= E_0:
            keep = np.random.uniform() < np.exp(B* (E_0 - E_f))
        else:
            keep = True

        if keep:
            E_0 = E_f
        else:
            lattice[site1], lattice[site2] = lattice[site2], lattice[site1]

    return np.array([correlation(lattice, r) for r in xrange(1, N/2+1)])

#TODO should this return tuples?
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

def energy(lattice):
    '''
    Calculate the energy of a lattice
    :param lattice:
        Lattice to calculate the energy on
    :param B:
        Float, inverse temp times boltzman factor
    :return:
        Energy of the lattice
    '''
    t0 = time()
    Vn = nuclear_potential(lattice)
    t1 = time()
    Vc = coulomb_potential(lattice)
    t2 = time()
    #print 'Vn time: %.5f s \t Vc time: %0.5f s'%(t1-t0, t2-t1)
    #print Vn, Vc
    return Vn+Vc 


def nuclear_potential(lattice):
    '''
    Potential based on strong force interatcions
    :param lattice :
        Lattice of (-1, 0, 1) sites to compure NN strong force interactions on
    :param B:
        Float, Inverse temp times boltzman constant
    :return:
        Strong force interaction energy of entire lattice at B
    '''

    N = lattice.shape[0]

    V = 0
    for site in site_idxs['occ']:
        spin_site = lattice[site]
        nn = get_NN(site, N, d)
        for neighbor in nn:
            spin_nn = lattice[tuple(neighbor)]
            if spin_nn == 0:
                continue
            V += v[(spin_site, spin_nn)]
    
    return V
def coulomb_potential(lattice):
    """
    The potential from the
    :param lattice:
        Lattice to calculate statistic on
    :return:
    """
    N = lattice.shape[0]
    V0 = 0 #TODO figure out the value of this
    e = 1.6e-19 #TODO this isn't right either!
    v0 = (e**2)/(a*N)

    V = V0
    #TODO numpy where != 0
    #TODO can keep a list of proton, neutrons and unoccupied sites
    visited_sites = set()
    dimpot_opps = 0
    while site_idxs[1]:
        site = site_idxs[1].pop()
        visited_sites.add(site)
        t0 = time()
        #avoid double counting by only iterating over remaining elements
        for neighbor in site_idxs[1]:
            V+=v0*dimensionless_potential(site, neighbor, N)
            dimpot_opps+=1
        t1 = time()
        #print 'Col Loop Time: %.3f s'%(t1-t0)

    #print dimpot_opps
    site_idxs[1] = visited_sites
    return V

#TODO precomputation
def dimensionless_potential(site1, site2, N):
    """

    :param site1:
        d-tuple of a coordinate in the lattice
    :param site2:
        d-tuple of a coordinate in the lattice
    :return:
        The value of the potential between these two sites.
    """
    diff = tuple([(site2[i]-site1[i])for i in xrange(d)])
    dist = np.sqrt(np.sum(np.array(diff)**2))

    #TODO i'll wanna save this so all objects have access to it.
    #if round(dist, 1) not in dim_pot_dict:
    #    print diff, round(dist,1)

    if round(dist,1) in dim_pot_dict:
        return dim_pot_dict[round(dist,1)]

    s_0 = a_s*1.0/N
    #TODO this term is basically inconsequential!
    u_sr = erfc(dist/s_0)/dist

    u_lr = 0
    for l in product(range(L_MAX), repeat = d):
        if l == tuple(0 for i in xrange(d)):
            continue
        l2 = sum(l_i for l_i in l)
        u_lr += (np.exp(-1*np.pi**2*s_0**2*l2)/(np.pi*l2) )*(np.exp(-2*np.pi*1j*np.dot(np.array(l), diff))).real
        #print u_lr, l

    #print dist, u_sr, u_lr
    u_tot = u_sr+u_lr
    dim_pot_dict[round(dist,1)] = u_tot
    return u_sr + u_lr


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

    xi = 0.0
    for site in all_sites:
        nn = get_NN(site, N, d, r)
        for neighbor in nn:
            xi += 1.0 if lattice[site]==lattice[tuple(neighbor)] else -1.0

    return xi/((N**d)*d)

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an ising model')
    parser.add_argument('N', type = int, help = 'Length of one side of the cube.')
    #parser.add_argument('B', type = float, help ='Bond coupling strength.')

    parser.add_argument('xb', type = float, default = 1.0, nargs = '?',\
                        help = 'Baryon occupation fraction. Default is 1.0')
    parser.add_argument('xp', type = float, default = 0.5, nargs = '?',
                        help = 'Proton fraction. Default is 0.5')
    parser.add_argument('n_steps', type = int, default = 1000, nargs = '?',\
                        help = 'Number of steps to simulate. Default is 1e5')
    parser.add_argument('--plot', action = 'store_true',\
                        help = 'Whether or not to plot results. Only allowed with d = 1 or 2.')

    args = parser.parse_args()
    spins = []
    Bs = [0.001]#,  0.1, 1.0, 10]
    for B in Bs:
        print B
        spins.append(run_ising(B = B, **vars(args)))

    for B, spin in izip(Bs, spins):
        print B, spin
    #    plt.plot(spin, label = B )
    #plt.legend(loc = 'best')
    #plt.ylim([-0.1, 1.1])
    #plt.show()
