'''
This module contains functions that analyze clusters that form in the charged ising model.
'''

from collections import defaultdict
from itertools import izip
import numpy as np
from charged_ising import get_NN, pbc_r


def find_clusters(lattice):
    """
    Find clusters in the lattice
    :param lattice:
        np array of (-1, 0, 1) sites.
    :return clusters, occ_set
        clusters is a dictionary where the keys are the indicies and the values are sets containing 
        other members of that cluster
        occ_set is the set of all occupied sites, used in toher computations so is efficient to return.
    """
    #find clusters with disjoint sets
    disjoint_sets = defaultdict(set)

    occupied_sites = zip(*(lattice != 0).nonzero())

    #begin, each site is its own cluster
    for coord in occupied_sites:
        disjoint_sets[coord].add(coord)

    occ_set = set(occupied_sites)

    #for each site, get its nn and join clusters
    for site in occupied_sites:
        nn = set(get_NN(site, lattice.shape[0], len(lattice.shape), full=True)) & occ_set
        # this is an nn that is occupied! we have structure.
        for neighbor in nn:
            new_set = disjoint_sets[site] | disjoint_sets[neighbor]
            
            for ns in new_set:
                disjoint_sets[ns] = new_set

    return disjoint_sets, occ_set


def unique_clusters(clusters):
    """
    Find unique clusters from a cluster dict returned from find_clusters
    :param clusters:
        dictionary returned from find clusters linking sites to their clusters
    :return uniques:
        a list of all unique clusters in clusters
    """
    uniques = []
    all_clusters = set()
    for cluster in clusters.itervalues():
        if not all_clusters >= cluster:
            all_clusters |= cluster
            uniques.append(cluster)

    return uniques


def pbc_com(cluster, lattice):
    """
    Center of mass in periodict boundary conditions. Found this on wikipedia.
    :param cluster:
        A set containing the d-tuples that comprise a unique cluster
    :param lattice:
        The lattice the cluster lives on. 
    :return pbc_com
        np array of shape (d,) with the COM of the cluster in PBC
    """
    coords = np.array(list(cluster), dtype=float) / lattice.shape[0]
    xis, zetas = np.zeros_like(coords), np.zeros_like(coords)
    for i, c in enumerate(coords):
        xis[i, :] = np.cos(c * 2 * np.pi)
        zetas[i, :] = np.sin(c * 2 * np.pi)

    xibar = xis.mean(axis=0)
    zetabar = zetas.mean(axis=0)

    theta_bar = np.arctan2(-zetabar, -xibar) + np.pi

    return theta_bar * lattice.shape[0] / (2 * np.pi)


def cluster_size(cluster, lattice, occ_set):
    """
    Radial size of cluster. Computed by calculating the average distance from boundary particles to the COM
    :param cluster:
        A set containing the d-tuples that comprise a unique cluster
    :param lattice:
        The lattice the cluster lives on. 
    :param occ_set:
        Set continaing all occupied site indicies in the lattice. Returned from find_clusters
    :return avg_size
        float, average radial size of the cluster
    """
    COM = pbc_com(cluster, lattice)

    d = len(lattice.shape)
    N = lattice.shape[0]
    
    edge_sites = []
    for site in map(tuple, cluster):
        nn = get_NN(site, N,d, full=True)
        #not all NN are occupied; is an edge site
        if not all(neighbor in occ_set for neighbor in nn):
            edge_sites.append(site)

    avg_dist = np.mean(
        [np.linalg.norm(np.array(pbc_r(site, COM,N )) * N) for site in edge_sites])

    return avg_dist


def cluster_moments(cluster, lattice):
    """
    Calculate the moments of inertia of a cluster around d major axes, normalized by cluster size.
    param cluster:
        A set containing the d-tuples that comprise a unique cluster
    :param lattice:
        The lattice the cluster lives on. 
    :return moments
        a length d numpy array with the moments of inertia of the cluster 
    """
    coords = np.array(list(cluster), dtype=float)
    COM = pbc_com(cluster, lattice)

    d = len(lattice.shape)
    N = lattice.shape[0]
    moments = np.zeros((d,))

    for i in xrange(d):
        for j in xrange(d):
            if i == j:
                continue
            moments[i] += np.sum((pbc_r(c, COM, N)[j] * N) ** 2 for c in coords)

    return moments / coords.shape[0]
