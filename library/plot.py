import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import candidates

def plotEvents(data, fs, events, labels=None, colored=[]):
    # Plot events

    plt.figure(figsize=(16, 6))
    nEvents = min(25, len(events))

    for i in range(nEvents):
        t0 = int((events[i][0] - 0.3) * fs)
        t1 = int(t0 + fs)
        t = (np.arange(t0, t1) - t0) / fs + i

        plt.axvline(i, color="gray")

        for j in range(data.shape[0]):
            d = data[j, t0:t1]
            m = np.mean(d)
            if j in colored:
                plt.plot(t, (d - m) - 50 * (j), "b")
            else:
                plt.plot(t, (d - m) - 50 * (j), "k")

    if labels:
        plt.yticks(
            [-50 * (j) for j in range(data.shape[0])],
            labels,
        )
    else:
        plt.yticks(
            [-50 * (j) for j in range(data.shape[0])],
            ["EEG {}".format(j) for j in range(data.shape[0])],
        )
    plt.xticks([0.5 + i for i in range(nEvents)], [i for i in range(nEvents)])
    plt.xlim(0, nEvents)
    plt.show()


def plotAvgIed(data, fs, events):
    # Average IED
    dataAvg = list()
    for j in range(data.shape[0]):
        dataAvg.append(list())

    for i in range(len(events)):
        t0 = int((events[i][0] - 0.3) * fs)
        t1 = t0 + int(fs)
        for j in range(data.shape[0]):
            dataAvg[j].append(data[j, t0:t1] - np.mean(data[j, t0:t1]))

    dataAvg = np.mean(dataAvg, axis=1)

    plt.figure(figsize=(1, 4))
    plt.title("Avg IED")

    t = np.arange(fs) / fs

    for j in range(data.shape[0]):
        plt.plot(t, dataAvg[j] - 100 * (j), "k")

    plt.yticks(
        [-100 * (j) for j in range(data.shape[0])],
        ["EEG {}".format(j) for j in range(data.shape[0])],
    )
    plt.xlim(0, 1)
    plt.xlabel("Time [s]")
    plt.show()


def plotElectrodePairs(pairs):
    ## Top projection of electrodes
    cmap = matplotlib.cm.get_cmap('tab20')

    fig = plt.figure(figsize=(16,6))
    ax = fig.add_subplot(projection='3d')

    for c, elec in enumerate(pairs):
        X, Y, Z  = [], [], []

        for i in range(2):
            theta = elec[i][0]
            phi = elec[i][1]
            if np.abs(phi) > np.pi/2:
                lphi = phi - np.sign(phi)*np.pi/2
                x, y, z = candidates.spherical2cart(candidates.r*(1+np.cos(lphi)/5), theta, np.sign(phi)*np.pi/2)
            else:
                x, y, z = candidates.spherical2cart(candidates.r, theta, phi)
            ax.scatter(x, y, z, color=cmap(c%20))
            X.append(x)
            Y.append(y)
            Z.append(z)
        ax.plot(X, Y, Z, color=cmap(c%20))

    ax.view_init(elev = 90, azim = -90)

    dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
    dx, dy = dx.real, dx.imag
    nose_x = np.array([-dx, 0, dx]) * candidates.r
    nose_y = np.array([dy, 1.15, dy]) * candidates.r
    ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                        .532, .510, .489]) * (candidates.r * 2)
    ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                        -.1313, -.1384, -.1199]) * (candidates.r * 2)

    ax.plot(nose_x, nose_y, zs=0, zdir='z', color='black')
    ax.plot(ear_x, ear_y, zs=0, zdir='z', color='black')
    ax.plot(-ear_x, ear_y, zs=0, zdir='z', color='black')

    angle = np.linspace( 0 , 2 * np.pi , 150 ) 
    radius = candidates.r
    x = radius * np.cos( angle ) 
    y = radius * np.sin( angle )
    ax.plot(x, y, zs=0, zdir='z', color='black')

    plt.axis('off')

    return fig
