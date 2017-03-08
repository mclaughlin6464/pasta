'''
This module contains functions that analyze clusters that form in the charged ising model.
'''

from collections import defaultdict
from itertools import izip
import numpy as np
from charged_ising import get_NN, pbc_r


def find_clusters(lattice):
    disjoint_sets = defaultdict(set)

    occupied_sites = zip(*(lattice != 0).nonzero())

    for coord in occupied_sites:
        disjoint_sets[coord].add(coord)

    occ_set = set(occupied_sites)

    # uniques = len(occupied_sites)

    for site in occupied_sites:
        nn = set(get_NN(site, lattice.shape[0], len(lattice.shape), full=True)) & occ_set
        # this is an nn that is occupied! we have structure.
        for neighbor in nn:

            new_set = disjoint_sets[site] | disjoint_sets[neighbor]
            for ns in new_set:
                disjoint_sets[ns] = new_set

    return disjoint_sets, occ_set


def unique_clusters(clusters):
    uniques = []
    all_clusters = set()
    for cluster in clusters.itervalues():
        if not all_clusters >= cluster:
            if len(all_clusters & cluster) > 0:
                print all_clusters & cluster
            all_clusters |= cluster
            uniques.append(cluster)

    return uniques


def pbc_com(cluster, lattice):
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
    COM = pbc_com(cluster, lattice)
    # coords-=COM

    edge_sites = []
    for site in map(tuple, cluster):
        nn = get_NN(site, lattice.shape[0], len(lattice.shape), full=True)
        if not all(neighbor in occ_set for neighbor in nn):
            edge_sites.append(site)

    avg_dist = np.mean(
        [np.linalg.norm(np.array(pbc_r(site, COM, lattice.shape[0])) * lattice.shape[0]) for site in edge_sites])

    return avg_dist


def cluster_moments(cluster, lattice):
    coords = np.array(list(cluster), dtype=float)
    COM = pbc_com(cluster, lattice)
    # coords-=COM

    d = len(lattice.shape)
    N = lattice.shape[0]
    moments = np.zeros((d,))

    for i in xrange(d):
        # moments[i] = np.sum(int(j!=i)*np.sum(pbc_r(c, com, N)**2) for j in xrange(d))
        for j in xrange(d):
            if i == j:
                continue
            moments[i] += np.sum((pbc_r(c, COM, N)[j] * N) ** 2 for c in coords)

    # return moments/(2*np.sum(moments))
    return moments / coords.shape[0]