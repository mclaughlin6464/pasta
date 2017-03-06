'''
I'm going to modify my model for the simple ising model to the charged ising model, from Hasnaoui et al.
'''

import argparse
import os
from time import time
from itertools import izip, product
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from scipy.special import erfc
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
cmap = 'coolwarm'

d = 2  # fixed in this model
v = {(1, 1): 5.167, (-1, -1): 5.167, (1, -1): -5.5, (-1, 1): -5.5}  # MeV
a = 1.842e-15  # lattice spaccing in m
a_s = 1.0  # dimensionaless smoothing length
L_MAX = 10  # TODO try other values

downsample_rate = 1000

dim_pot_dict = {}  # store old results for the dimensionless potential

site_idxs = {}

pbc_r_dict = defaultdict(dict) # store distances

def run_ising(N, B_goal, xb, xp, n_steps, outputdir, plot=False):
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

    if plot:
        try:
            assert d <= 2
        except AssertionError:
            raise AssertionError("Can only plot in one or two dimensions.")

    assert 0 < N < 100
    assert B_goal > 0
    assert 0 <= xb <= 1
    assert 0 <= xp <= 1
    assert n_steps > 0

    np.random.seed(0)
    # initialize based on inputs
    size = tuple(N for i in xrange(d))
    lattice = (np.random.rand(*size) <= xb).astype(int)  # occupied
    lattice *= (2 * (np.random.rand(*size) <= xp) - 1).astype(int)  # neturon/proton fraction

    # make a list of indices
    for spin in (1, 0, -1):
        site_idxs[spin] = set(tuple(idx) for idx in np.c_[(lattice == spin).nonzero()])

    site_idxs['occ'] = site_idxs[1] | site_idxs[-1]  # sites occupied by nucleon
    E_0 = energy(lattice)

    if xb == 1.0 and (xp == 0.0 or xp == 1.0): #no swaps possible, will infinite loop
        print E_0/(N**d)
        energies = np.array([E_0 for i in xrange(n_steps+2)])
        return energies

    energies = np.zeros((n_steps+2,))
    energies[0] = E_0

    npoints = 1000
    euclid_r_avgs = np.zeros(((n_steps)/downsample_rate+1.0, npoints, 3))
    taxicab_r_avgs = np.zeros(((n_steps)/downsample_rate+1.0, npoints, 3))
    cvs = np.zeros(((n_steps)/downsample_rate+1.0,))

    if plot:
        plt.ion()

    tf, t0 = 0, 0

    B0 = 0.01 #start at high T
    B_goal_low = B0 < B_goal #if we have to go up or down

    B = B0
    for step in xrange(n_steps+1):

        if step % downsample_rate == 0:

            if step > n_steps / 4: #update temperature
                if B_goal_low:
                    if B < B_goal:
                        B *= 1.3
                    if B > B_goal:
                        B = B_goal
                else:
                    if B > B_goal:
                        B /= 1.3
                    if B < B_goal:
                        B = B_goal

            E_0 = energies[step]
            tf = time()
            Cv = heat_capacity(B, energies[step-n_steps/downsample_rate:step])
            cvs[step/downsample_rate] = Cv
            for i, particle in enumerate((-1, 1)):
                euclid_r_avgs[step/downsample_rate, :, i] = pair_correlation(particle, npoints, N)
                taxicab_r_avgs[step/downsample_rate, :, i] = pair_correlation(particle, npoints, N, dist='taxicab')

            np.save(os.path.join(outputdir, 'lattice_%3d.npy'), lattice)

            print step, E_0, E_0 / (xb * N ** d), B,Cv/len(site_idxs['occ']), tf - t0
            print

            if plot:
                if d == 1:
                    # im = plt.imshow(lattice.reshape((1, -1)), interpolation='none')
                    sns.heatmap(lattice.reshape((1, -1)), cbar=True, cmap=cmap, vmin = -1, vmax = 1)
                else:
                    # im = plt.imshow(lattice, interpolation='none')
                    sns.heatmap(lattice, cbar=True, cmap=cmap, vmin = -1, vmax = 1)
                # plt.colorbar(im)
                plt.title(r'$\beta= %e, E/A=%0.2f, C_v=%0.2f$' % (B, E_0 / (xb * N ** d), Cv/len(site_idxs['occ']) ) )
                plt.pause(0.1)
                plt.clf()

            t0 = time()

        site1, site2 = [tuple(0 for i in xrange(d)) for i in xrange(2)]
        while lattice[site1] == lattice[site2]:
            # consider flipping this site
            site1, site2 = np.random.randint(0, N, size=(2, d))
            site1, site2 = tuple(site1), tuple(site2)

        energies[step+1] = energies[step]
        dE = delta_energy(lattice, site1, site2)

        # if E_F < E_0, keep
        # if E_F > E_0, keep randomly given change of energies
        if np.random.uniform() < np.exp(B * (-dE)):
            # make the appropriate switches in the sets
            site_idxs[lattice[site1]].remove(site1)
            site_idxs[lattice[site2]].remove(site2)

            site_idxs[lattice[site1]].add(site2)
            site_idxs[lattice[site2]].add(site1)

            if lattice[site1] == 0:
                # site1 is becoming occupied and site2 is becoming unoccupiated
                site_idxs['occ'].add(site1)
                site_idxs['occ'].remove(site2)
            elif lattice[site2] == 0:
                # vice versa
                site_idxs['occ'].add(site2)
                site_idxs['occ'].remove(site1)
            lattice[site1], lattice[site2] = lattice[site2], lattice[site1]

            energies[step+1] += dE

    if plot:
        if d == 1:
            # im = plt.imshow(lattice.reshape((1, -1)), interpolation='none')
            sns.heatmap(lattice.reshape((1, -1)), cbar=True, cmap=cmap)
        else:
            # im = plt.imshow(lattice, interpolation='none')
            sns.heatmap(lattice, cbar=True, cmap=cmap)
        # plt.colorbar(im)
        plt.title(r'$\beta= %e, E=%0.2f$' % (B, E_0))
        # while True:
        #    plt.pause(0.1)
        plt.clf()

    euclid_correlations = np.zeros((npoints, 3))
    taxicab_correlations = np.zeros((npoints/10, 3))
    for i, particle in enumerate((-1, 1)):
        for j, r in enumerate( np.linspace(0, 1, npoints)):
            euclid_correlations[j,i] = g_r(particle, euclid_r_avgs[euclid_r_avgs.shape[0]/2:,j,i], N)

        for j, r in enumerate(np.linspace(0, 1, npoints/10)):
            taxicab_correlations[j,i] = g_r(particle, taxicab_r_avgs[taxicab_r_avgs.shape[0]/2:,j,i], N)

    return energies, cvs, euclid_correlations, taxicab_correlations


def get_NN(site, N, d, r=1, full=False):
    '''
    The NN of the site. Will only return those UP in index (east, south, and down) to avoid double counting.
    Accounts for PBC
    :param site:
        (d,) array of coordinates in the lattice
    :param N:
        Size of one side of the lattice
    :param d:
        dimension of the lattice
    :param r:
        int, optional. Distance to NN to return. Default is 1.
    :param full:
        bool, optional. Whether or not to return "backward" neighbors on top of "forward ones." Default is False.
    :return:
        dxd numpy array where each row corresponds to the nearest neighbors.
    '''
    mult_sites = np.r_[[site for i in xrange(d)]]
    adjustment = np.eye(d) * r

    output = map(tuple, ((mult_sites + adjustment) % N).astype(int) )
    if full:
        output.extend(map(tuple, ((mult_sites - adjustment + N) % N).astype(int) ))
    # return backwards neighbors also
    return output


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
    Vn = nuclear_potential(lattice)
    Vc = coulomb_potential(lattice)
    #print Vn, Vc
    return Vc#  +Vc


def delta_energy(lattice, site1, site2):
    """
    Calculate the change in energy from swapping these two sites. Assumes swap has not yet been performed.
    :param lattice:
        Lattice to calculate change on.
    :param site1:
        d-tuple, index of first site to swtich.
    :param site2:
        d-tuple, index of second site to switch
    :return:
        dE, the energy that would result if the sites were swapped.
    """
    dVn = delta_nucpot(lattice, site1, site2)
    dVc = delta_colpot(lattice, site1, site2)
    #print dVn, dVc
    return dVc# +dVc


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
    visited_sites = set()
    V = 0
    additions = 0
    while site_idxs['occ']:
        site = site_idxs['occ'].pop()
        visited_sites.add(site)
        spin_site = lattice[site]
        nn = set(get_NN(site, N, d)) & (site_idxs['occ'] | visited_sites)
        for neighbor in nn:
            spin_nn = lattice[tuple(neighbor)]
            V += v[(spin_site, spin_nn)]
            additions+=1

    site_idxs['occ'] = visited_sites

    return V


def delta_nucpot(lattice, site1, site2):
    '''
    The change in energy from swapping these two sites.
    :param lattice:
        Lattice of (-1, 0, 1 ) sites to compute NN strong force interactions on
    :param site1:
        d-tuple of first site in swap
    :param site2:
        d-tuple of second site in swap
    :return:
        dV, float, change in potential
    '''

    N = lattice.shape[0]

    V = 0
    for siteA, siteB in [(site1, site2), (site2, site1)]:
        spin_siteA = lattice[siteA]
        spin_siteB = lattice[siteB]
        nn = set(get_NN(siteA, N, d, full=True)) & site_idxs['occ']

        nn.discard(siteB) #make usre you don't count energy that's already there!

        for neighbor in nn:
            spin_nn = lattice[neighbor]
            if spin_siteA != 0:
                V -= v[(spin_siteA, spin_nn)]
            if spin_siteB != 0:
                V += v[(spin_siteB, spin_nn)]

    return V


def coulomb_potential(lattice):
    """
    The potential from the
    :param lattice:
        Lattice to calculate statistic on
    :return:
    """
    N = lattice.shape[0]
    e = 1.6e-19
    v0 = (9e9) * (e ** 2) / (a * N) * (6.242e12)

    #calculate constant
    Z = len(site_idxs[1])
    s_0 = a_s * 1.0 / N
    U = -1*(Z/(np.sqrt(np.pi)*s_0)+0.5*np.pi*s_0**2*Z**2)
    u_lr0 = 0
    for l in product(range(L_MAX), repeat=d):
        if l == tuple(0 for i in xrange(d)):
            continue
        l2 = sum(l_i ** 2 for l_i in l)
        u_lr0 += np.exp(-1 * np.pi ** 2 * s_0 ** 2 * l2)
    U+=Z/2*u_lr0

    #Had to do this cuz i was getting non-deterministic behavior
    proton_list = sorted(list(site_idxs[1]), key = lambda x: x[0])
    #print len(site_idxs[1])
    while proton_list:
        site = proton_list.pop()
        for neighbor in proton_list:
            U += dimensionless_potential(site, neighbor, N)

    return v0*U


def delta_colpot(lattice, site1, site2):
    """
    Change in electric potential energy from swapping the two sites.
    :param lattice:
        Lattice of (-1, 0, 1 ) sites to compute NN strong force interactions on
    :param site1:
        d-tuple, index of first site to swtich.
    :param site2:
        d-tuple, index of second site to switch
    :return:
        dV, float, the change in potential from the swap
    """
    spin1, spin2 = lattice[site1], lattice[site2]
    if (spin1, spin2) in [(0,0), (0, -1), (-1, 0), (-1,-1), (1,1)]:
        return 0
    N = lattice.shape[0]
    e = 1.6e-19
    v0 = (9e9) * (e ** 2) / (a * N) * (6.242e12)

    U = 0 #constant doesn't matter here

    pos_site = site1 if spin1 == 1 else site2
    zero_site = site2 if spin1 == 1 else site1

    site_idxs[1].remove(pos_site)

    for neighbor in site_idxs[1]:
        U -= dimensionless_potential(pos_site, neighbor, N)
        U += dimensionless_potential(zero_site, neighbor, N)

    site_idxs[1].add(pos_site)

    return U*v0

def pbc_r(site1, site2, N):
    """
    The distance vector between two sites in PBC
    :param site1:
        d- tuple of the location of the first site
    :param site2:
        d- tuple of location of the second site
    :param N:
        Sidelength of one side of the lattice
    :return:
        r, d-tuple, distance between two points
    """
    diff = tuple([min(abs(float(site2[i] - site1[i]) / N), abs(1-float(site2[i] - site1[i]) / N) ) for i in xrange(d)])

    return diff


def dimensionless_potential(site1, site2, N):
    """

    :param site1:
        d-tuple of a coordinate in the lattice
    :param site2:
        d-tuple of a coordinate in the lattice
    :return:
        The value of the potential between these two sites.
    """
    diff = pbc_r(site1, site2, N)
    dist = np.sqrt(np.sum(np.array(diff) ** 2))

    if round(dist, 1) in dim_pot_dict:
        return dim_pot_dict[round(dist, 1)]

    s_0 = a_s * 1.0 / N
    # s_0 = 0.12
    # TODO this term is basically inconsequential!
    u_sr = erfc(dist / s_0) / dist

    u_lr = 0
    for l in product(range(L_MAX), repeat=d):
        if l == tuple(0 for i in xrange(d)):
            continue
        l2 = sum(l_i ** 2 for l_i in l)

        u_lr += (np.exp(-1 * np.pi ** 2 * s_0 ** 2 * l2) / (np.pi * l2)) * (
        np.exp(-2 * np.pi * 1j * np.dot(l, diff))).real
    #print site1, site2, u_lr
    u_tot = u_sr + u_lr
    dim_pot_dict[round(dist, 1)] = u_tot
    return u_tot


def heat_capacity(B, Es):
    """
    Calculate the heat capacity of the lattice.
    :param B:
        Float. Beta, inverse temperature parameter.
    :param E:
        Np array of floats. Total energy of the lattice measured at several times.
    :return:
        Cv, the heat capacity.
    """
    #Bf = 1.0/11700
    #Cv =  3/2*len(site_idxs['occ'])+np.pi**2*len(site_idxs[1])*(Bf/B)+
    Cv = (B**2)*Es.var()
    return Cv

def _pair_correlation(particle, r, N, tol = 1e-3):
    """
    Correlation between two particles or spaces at a given distance.
    :param particle:
        type of particle. Must be among {-1, 0, 1, "occ"}, where occ is all nucleons
    :param r:
        float, Distance to count particles at. Must be between 0 and 1.
    :param N:
        int, size of the lattice
    :param tol:
        float, distance tolerance. Default is 1e-3
    :return:
        r_avg, the pair-pair correlation of this particular particle on the lattice.
    """
    assert particle in site_idxs
    assert 0 <= r <= 1

    r_avg = 0
    paircount = 0
    Np = len(site_idxs[particle])
    visited_sites = set()

    while site_idxs[particle]:
        site1 = site_idxs[particle].pop()
        visited_sites.add(site1)

        for site2 in site_idxs[particle]:

            r_vec = pbc_r(site1, site2, N)
            r_mag = np.sqrt(sum([r_i**2 for r_i in r_vec]))
            paircount+=1
            if np.abs(r-r_mag) < tol:
                r_avg += 1

    site_idxs[particle] = visited_sites

    if paircount == 0:
        return 1

    #return 1 + N**d/(Np*(Np-1.0))*(r_avg*1.0/paircount)
    return r_avg

def pair_correlation(particle, npoints, N, dist = 'euclidean'):
    """
    Correlation between two particles or spaces at a given distance.
    :param particle:
        type of particle. Must be among {-1, 0, 1, "occ"}, where occ is all nucleons
    :param r:
        float, Distance to count particles at. Must be between 0 and 1.
    :param N:
        int, size of the lattice
    :param tol:
        float, distance tolerance. Default is 1e-3
    :return:
        r_avg, the pair-pair correlation of this particular particle on the lattice.
    """
    assert particle in site_idxs
    assert dist in {'euclidean', 'taxicab'}
    if len(site_idxs[particle]) == 0:
        return np.histogram(np.zeros((d,)),np.linspace(0,1,num=npoints+1))[0]

    normpos = np.array(list(site_idxs[particle]), dtype=float)#/N
    diffs = np.zeros( ((normpos.shape[0]*(normpos.shape[0]-1))/2, d) )
    idxs = np.triu_indices(normpos.shape[0] , 1)
    for dim in xrange(d):
        x = np.reshape(normpos[:,dim], (len(normpos), 1))
        diffmat =  x - x.transpose()
        diffs[:,dim] = np.abs(diffmat[idxs].flatten())

    pbc_diffs = np.where(diffs > N/2, N-diffs, diffs)
    if dist == 'euclidean':
        pbc_dists = norm(pbc_diffs, axis = 1)
    else:
        pbc_dists = np.sum(pbc_diffs, axis = 1)

    return np.histogram(pbc_dists,np.linspace(0,1*N,num=npoints+1))[0]

def g_r(particle, r_avgs,N):
    '''
    Calculate g_r from many measurements of r_avg
    :param particle:
        type of particle. Must be among {-1, 0, 1, "occ"}, where occ is all nucleons
    :param r_avgs:
        np array of calculations of pair_correlation
    :param N:
        sidelength of the lattice
    :return:
        g(r), the pair-pair  correlation function
    '''

    Np = len(site_idxs[particle])
    if Np == 0:
        return  1
    return 1 + N**d/(Np*(Np-1.0))*(r_avgs.mean())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an ising model')
    parser.add_argument('N', type=int, help='Length of one side of the cube.')
    parser.add_argument('B_goal', type=float, help='Bond coupling strength.')

    parser.add_argument('xb', type = float, default = 1.0, nargs = '?',\
                        help = 'Baryon occupation fraction. Default is 1.0')
    parser.add_argument('xp', type=float, default=0.5, nargs='?',
                        help='Proton fraction. Default is 0.5')
    parser.add_argument('n_steps', type=int, default=1000, nargs='?', \
                        help='Number of steps to simulate. Default is 1e5')
    parser.add_argument('outputdir', type=str, default='./', nargs='?', \
                        help='Directory to save outputs. Default is current directory.')
    parser.add_argument('--plot', action='store_true', \
                        help='Whether or not to plot results. Only allowed with d = 1 or 2.')

    args = parser.parse_args()

    if not os.path.isdir(args.outputdir):
        os.mkdir(args.outputdir)

    energies = []
    #xbs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #for xb in xbs:
        #print xb
        #energies.append(run_ising(xb=xb, **vars(args)))

    energies,cvs, euclid_correlations, taxicab_correlations = run_ising(**vars(args))

    print energies[energies.shape[0]/2:].mean()/(len(site_idxs['occ']))
    print cvs[cvs.shape[0]/2:].mean()

    plt.plot(energies/len(site_idxs['occ']), label = 'E')
    plt.plot([energies[i:i+100].var()/len(site_idxs['occ']) for i in xrange(energies.shape[0]-100)], label = 'C')
    plt.legend(loc = 'best')

    #plt.savefig(os.path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_energies.png'%(args.xb,args.xp)))

    np.savetxt(os.path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_energies.npy'%(args.xb,args.xp)), energies, delimiter=',')
    np.savetxt(os.path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_energies.npy'%(args.xb,args.xp)), energies, delimiter=',')
    plt.clf()

    #if not args.plot:
    #    plt.ion()
    #while True:
    #    plt.pause(0.1)

    rs = args.N*np.linspace(0,1, euclid_correlations.shape[0])
    plt.plot(rs, euclid_correlations[:,0], label = 'Neutrons')
    plt.plot(rs, euclid_correlations[:,1], label = 'Protons')
    #plt.vlines([1, np.sqrt(2), 2], 1, 2)
    #plt.plot(rs, correlations[:,2], label = 'Nucleons')
    plt.legend(loc = 'best')

    while True:
        plt.pause(0.1)

    plt.savefig(os.path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_correlations.png'%(args.xb,args.xp)))

    np.savetxt(os.path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_euclid_correlations.npy'%(args.xb,args.xp)), euclid_correlations, delimiter=',')
    np.savetxt(os.path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_taxicab_correlations.npy'%(args.xb,args.xp)), taxicab_correlations, delimiter=',')


        # for xb, E in izip(xbs, energies):
    #plt.plot(xbs, energies)
    # plt.ylim([-0.1, 1.1])
    #plt.savefig('/home/sean/GitRepos/pasta/files/GroundStates.png')
    #while True:
    #    plt.pause(0.1)
