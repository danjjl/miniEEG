"""
Functions defining head geometry and generation of electrode candidate pool.

This module assumes mathematics spherical coordinate system.
"""

from math import radians
from numba import cfunc, njit, prange
import numpy as np
from numpy.linalg import norm

cirucumference = 57  # Average head circumference in cm
r = cirucumference / (2 * np.pi)


def loadElectrodeLocations(locationFile):
    """Load electrode positions stored in a text file.

    This function expects an txt style txt file where lines are of the format
    E"ElecNum"\t"theta(degrees)"\t"phi(degrees)"
    e.g.
    E25	64.57	-35.04
    The function returns a list of electrode positions as a tuples (theta, phi)
    in radians.

    Args:
        locationFile: location of ELP file
    Return:
        electrodes: list of (theta, phi) electrode positions in radians
    """
    electrodes = list()
    with open(locationFile, "r") as f:
        for line in f.readlines()[1:]:
            field = line.split("	")
            electrodes.append(
                (radians(float(field[-1])), radians(float(field[-2])))
            )
    return electrodes


def spherical2cart(r, theta, phi):
    """Convert spherical to cartesian coordinates."""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return x, y, z


def cart2spherical(x, y, z):
    """Convert cartesian coordinates to spherical coordinates."""
    r = norm((x, y, z))
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)

    return r, theta, phi


def midpointSphere(s1, s2):
    """Find midpoint on the surface of a sphere
    
    Args:
        s1: (r, theta, phi)
        s2: (r, theta, phi)
    Return:
        s: (r, theta, phi)
    """
    x1 = spherical2cart(*s1)
    x2 = spherical2cart(*s2)

    x = np.mean((x1, x2), axis=0)
    xScale = x * s1[0] / norm(x)

    s = cart2spherical(*xScale)

    return (s1[0], *s[1:])

def vectorPair(s1, s2):
    x1 = spherical2cart(*s1)
    x2 = spherical2cart(*s2)

    d = np.array(x2) - np.array(x1)
    # d = d[:2]
    d /= np.linalg.norm(d)

    return d


@cfunc(
    "types.double(types.UniTuple(types.double, 2), types.UniTuple(types.double, 2), types.double)"
)
def haversine_distance(s1, s2, r):
    dist = (
        2
        * r
        * np.arcsin(
            np.sqrt(
                np.sin(0.5 * (s1[1] - s2[1])) ** 2
                + np.cos(np.pi / 2 - s1[1])
                * np.cos(np.pi / 2 - s2[1])
                * np.sin(0.5 * (s2[0] - s1[0])) ** 2
            )
        )
    )
    return dist


@njit(parallel=True)
def computeCoverage(electrodes, r):
    EVALP = 500
    evalTheta = np.linspace(-np.pi / 2, np.pi / 2, EVALP)
    evalPhi = np.linspace(-np.pi * (2 / 3), np.pi * (2 / 3), EVALP)

    coverageM = np.ones((EVALP, EVALP)) * 2 * np.pi * r

    for t in prange(len(electrodes)):
        elec = electrodes[t]
        for i, theta in enumerate(evalTheta):
            for j, phi in enumerate(evalPhi):
                # Haversine distance
                dist = haversine_distance(elec, (theta, phi), r)
                if dist < coverageM[i, j]:
                    coverageM[i, j] = dist
    return coverageM


def getNeighbourDist(r, electrodes, directions):
    neighbourhood = np.zeros((len(electrodes), len(electrodes)))

    for i, elec0 in enumerate(electrodes):
        for j, elec in enumerate(electrodes[i:]):
            j += i
            distance = haversine_distance(elec0, elec, r)
            distance = np.e ** ((2 / 3) * -distance)
            distance *= np.abs(directions[i] @ directions[j])
            neighbourhood[i, j] = distance
            neighbourhood[j, i] = neighbourhood[i, j]
    return neighbourhood


@njit
def getElecPairsWithDist(electrodes, m):
    pairs = list()
    pairsI = list()
    std = 0.25
    for i in prange(len(electrodes)):
        elec0 = electrodes[i]
        for j in range(i, len(electrodes)):
            elec = electrodes[j]
            distance = haversine_distance(elec0, elec, r)
            if m - std < distance < m + std:
                pairs.append((elec0, elec))
                pairsI.append((i, j))

    return np.array(pairs), np.array(pairsI)


def selectNPairs(n, pairs):
    midpoints = list()
    direction = list()
    for pair in pairs:
        midpoints.append(midpointSphere((r, *pair[0]), (r, *pair[1]))[1:])
        direction.append(vectorPair((r, *pair[0]), (r, *pair[1])))
    midpoints = np.array(midpoints)
    direction = np.array(direction)
    selection = [i for i in range(len(midpoints))]
    neighbourhood = getNeighbourDist(r, midpoints, direction)
    while len(selection) > n:
        densestI = np.argmax(np.sum(neighbourhood[selection][:, selection], axis=0))
        del selection[densestI]
    return np.array(selection)
