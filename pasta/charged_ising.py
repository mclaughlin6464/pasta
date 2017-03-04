'''
I'm going to modify my model for the simple ising model to the charged ising model, from Hasnaoui et al.
'''

import argparse
import path
from time import time
from itertools import izip, product
import numpy as np
from scipy.special import erfc
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()
# sns.set_style("white")
# sns.set_palette(sns.color_palette("BrBG", 3))
cmap = 'coolwarm'

d = 3  # fixed in this model
v = {(1, 1): 5.167, (-1, -1): 5.167, (1, -1): -5.5, (-1, 1): -5.5}  # MeV
a = 1.842e-15  # lattice spaccing in m
a_s = 1.0  # dimensionaless smoothing length
L_MAX = 10  # TODO try other values

dim_pot_dict = {}  # store old results for the dimensionless potential

site_idxs = {}


def run_ising(N, B, xb, xp, n_steps, outputdir, plot=False):
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

    # TODO output writing
    # TODO autocorrelation sampling
    # TODO high temp burn-in

    if plot:
        try:
            assert d <= 2
        except AssertionError:
            raise AssertionError("Can only plot in one or two dimensions.")

    # TODO wrap these better
    assert 0 < N < 100
    assert B > 0
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
    if plot:
        plt.ion()
    tf, t0 = 0, 0
    for step in xrange(n_steps+1):
        if step % (n_steps/100) == 0:
            if n_steps/2 >step > n_steps / 6:
                B *= 1.5
            #E_0 = energy(lattice)
            E_0 = energies[step]
            tf = time()

            Cv = heat_capacity(B, energies[step-n_steps/100:step])
            print Cv/len(site_idxs['occ'])
            print step, E_0, E_0 / (xb * N ** d), B, tf - t0
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
        # print step
        site1, site2 = [tuple(0 for i in xrange(d)) for i in xrange(2)]
        while lattice[site1] == lattice[site2]:
            # consider flipping this site
            site1, site2 = np.random.randint(0, N, size=(2, d))
            site1, site2 = tuple(site1), tuple(site2)

        # t0 = time()

        energies[step+1] = energies[step]
        dE = delta_energy(lattice, site1, site2)

        # t1 = time()
        # print 'Total calc time: %.3f s'%(t1-t0)
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
            #print dE, lattice[site1], lattice[site2]
            #print energies[step], energies[step+1], energy(lattice)
            #print '*'*30

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

    # return np.array([correlation(lattice, r) for r in xrange(1, N/2+1)])
    return energies


# TODO should this return tuples?
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
    t0 = time()
    Vn = nuclear_potential(lattice)
    t1 = time()
    Vc = coulomb_potential(lattice)
    t2 = time()
    # print 'Vn time: %.5f s \t Vc time: %0.5f s'%(t1-t0, t2-t1)
    # print Vn, Vc
    return Vn  +Vc


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
    # TODO this doesn't need E_0, just can calculate the delta.
    dVn = delta_nucpot(lattice, site1, site2)
    dVc = delta_colpot(lattice, site1, site2)
    return dVn +dVc


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
        nn = set(get_NN(site, N, d)) & site_idxs['occ']
        for neighbor in nn:
            spin_nn = lattice[tuple(neighbor)]
            if spin_nn == 0:
                continue
            V += v[(spin_site, spin_nn)]

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

    visited_sites = set()
    dimpot_opps = 0
    while site_idxs[1]:
        site = site_idxs[1].pop()
        visited_sites.add(site)
        t0 = time()
        # avoid double counting by only iterating over remaining elements
        for neighbor in site_idxs[1]:
            U += dimensionless_potential(site, neighbor, N)
            dimpot_opps += 1
        t1 = time()
        # print 'Col Loop Time: %.3f s'%(t1-t0)

    # print dimpot_opps
    site_idxs[1] = visited_sites
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
    dimpot_opps = 0

    pos_site = site1 if spin1 == 1 else site2
    zero_site = site2 if spin1 == 1 else site1

    site_idxs[1].remove(pos_site)

    for neighbor in site_idxs[1]:
        U -= dimensionless_potential(pos_site, neighbor, N)
        U += dimensionless_potential(zero_site, neighbor, N)
        dimpot_opps += 2

        # print 'Col Loop Time: %.3f s'%(t1-t0)
    site_idxs[1].add(pos_site)

    # print dimpot_opps

    return U*v0


def dimensionless_potential(site1, site2, N):
    """

    :param site1:
        d-tuple of a coordinate in the lattice
    :param site2:
        d-tuple of a coordinate in the lattice
    :return:
        The value of the potential between these two sites.
    """
    diff = tuple([float(site2[i] - site1[i]) / N for i in xrange(d)])
    dist = np.sqrt(np.sum(np.array(diff) ** 2))

    # TODO i'll wanna save this so all objects have access to it.
    # if round(dist, 1) not in dim_pot_dict:
    #    print diff, round(dist,1)

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
        # print u_lr, l

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
    Bf = 1 #TODO get right value of this.
    #Cv =  3/2*len(site_idxs['occ'])+np.pi**2*len(site_idxs[1])*(Bf/B)+ B*(np.mean(Es**2)-np.mean(Es)**2)
    #return Cv
    return B**2*Es.var()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate an ising model')
    parser.add_argument('N', type=int, help='Length of one side of the cube.')
    parser.add_argument('B', type=float, help='Bond coupling strength.')

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
    energies = []
    #xbs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #for xb in xbs:
        #print xb
        #energies.append(run_ising(xb=xb, **vars(args)))

    energies = run_ising(**vars(args))

    plt.plot(energies/len(site_idxs['occ']), label = 'E')
    #plt.plot([energies[i:i+100].var()/len(site_idxs['occ']) for i in xrange(energies.shape[0]-100)], label = 'C')
    plt.legend(loc = 'best')

    plt.savefig(path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_energies.png'%(args.xb,args.xp)))

    np.savetxt(path.join(args.outputdir, 'xb_%0.2f_xp_%0.2f_energies.npy'%(args.xb,args.xp)), energies, delimiter=',')

    #while True:
    #    plt.pause(0.1)

    # for xb, E in izip(xbs, energies):
    #plt.plot(xbs, energies)
    # plt.ylim([-0.1, 1.1])
    #plt.savefig('/home/sean/GitRepos/pasta/files/GroundStates.png')
    #while True:
    #    plt.pause(0.1)
