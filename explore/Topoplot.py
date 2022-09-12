# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import sys


subject = 'patient025'
distance = 5.0
numNodes = 7

params = dict()
params["IEDDURATION"] = 0.2

params["distance"] = distance
params["numNodes"] = numNodes
params["elecPoolSize"] = 256

params["fs"] = 100
params["fsDown"] = 20
params["fsClassify"] = 2
params["lag"] = int(params["IEDDURATION"] * params["fs"])
params["lagDown"] = int(params["IEDDURATION"] * params["fsDown"])

params["NFOLDS"] = 4

# Data
params["dataFiles"] = dict()
params["dataFiles"][
    "ROOT_EDF"
] = "/esat/biomeddata/jdan/HDEEG/edf"
params["dataFiles"][
    "ROOT"
] = "/users/sista/jdan/miniEEG"
params["dataFiles"]["subject"] = subject
params["dataFiles"]["edfFile"] = os.path.join(
    params["dataFiles"]["ROOT_EDF"],
    params["dataFiles"]["subject"] + ".edf"
)
params["dataFiles"]["eventFile"] = os.path.join(
    params["dataFiles"]["ROOT"], "Persyst",
    params["dataFiles"]["subject"] + ".xlsx"
)
params["dataFiles"]["locationFile"] = os.path.join(
    params["dataFiles"]["ROOT"],
    "electrodes.txt"
)
params['BASESAVE'] = "simulations/variableDistance/results"
params['BASESAVE'] = os.path.join(params["dataFiles"]["ROOT"],
                                  params['BASESAVE'],
                                  params["dataFiles"]["subject"])

# +
import os
import pickle
import sys

import mne
import numba as nb
import numpy as np
import scipy.signal
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.append("../library")
import candidates
from forwardGreedySelection import forwardGreedySearchSVD
import IedDetector as ied
import loading
import spir

# +
from IPython.display import display, Markdown

import plot
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

import pandas as pd
from sklearn.metrics import confusion_matrix

# +
results = dict()

#################
### Load Data ###
electrodes = candidates.loadElectrodeLocations(params["dataFiles"]["locationFile"])

pairs, pairsI = candidates.getElecPairsWithDist(
    nb.typed.List(electrodes), params["distance"]
)
elecPool = candidates.selectNPairs(params["elecPoolSize"], pairs)

# Load Data in MNE structure
data = loading.loadHDData(
    params["dataFiles"]["edfFile"],
    params["dataFiles"]["eventFile"],
    params["dataFiles"]["locationFile"],
    params["fs"],
)

# Mne Data -> Numpy Data
npData = data.get_data() * 1e6
npData = loading.reRef(npData, pairsI[elecPool])

# Remove non-physiological data
power = spir.rolling_rms(npData, int(params["fs"] / 2))
highPower = np.where(np.any(power > 100, axis=0))[0]
for i in highPower:
    i0 = max(0, i-int(params["fs"] / 1))
    i1 = min(npData.shape[1], i+int(params["fs"] / 1))
    npData[:, i0:i1] = 0
amplitude = np.abs(npData)
highAmp = np.where(np.any(amplitude > 200, axis=0))[0]
for i in highAmp:
    i0 = max(0, i-int(params["fs"] / 1))
    i1 = min(npData.shape[1], i+int(params["fs"] / 1))
    npData[:, i0:i1] = 0

import gc
del power, amplitude
gc.collect()
# Remove non-physiological annotations
events = loading.loadEventFile(params["dataFiles"]["eventFile"], data.info["sfreq"])
toKeep = []
for i, event in enumerate(events):
    if np.any(npData[:, event[0]]):
        toKeep.append(i)

mapping = {1: "IED"}
annot_from_events = mne.annotations_from_events(
    events=events[toKeep],
    event_desc=mapping,
    sfreq=data.info["sfreq"],
    orig_time=data.info["meas_date"],
)
data.set_annotations(annot_from_events)

# Downsample
dataDown = scipy.signal.decimate(npData, int(params["fs"] / params["fsDown"]))

results["pairs"] = pairs
results["pairsI"] = pairsI
results["elecPool"] = elecPool
# -

display(Markdown('### {} events'.format(len(data.annotations))))

mapping = {'IED': 1}
epochs = mne.Epochs(data,
                    events=events[toKeep],
                    tmin=-0.5,
                    tmax=0.5,
                    event_id=mapping,
                    preload=True)
evoked = epochs['IED'].average()

for subject in ['patient003', 'patient004', 'patient009', 'patient013', 'patient020', 'patient021', 'patient025', 'patient030', 'patient036', 'patient037', 'patient040', 'patient047', 'patient049', 'patient050', 'patient051', 'patient080']:

    distance = 5.0
    numNodes = 7

    params = dict()
    params["IEDDURATION"] = 0.2

    params["distance"] = distance
    params["numNodes"] = numNodes
    params["elecPoolSize"] = 256

    params["fs"] = 100
    params["fsDown"] = 20
    params["fsClassify"] = 2
    params["lag"] = int(params["IEDDURATION"] * params["fs"])
    params["lagDown"] = int(params["IEDDURATION"] * params["fsDown"])

    params["NFOLDS"] = 4

    # Data
    params["dataFiles"] = dict()
    params["dataFiles"][
        "ROOT_EDF"
    ] = "/esat/biomeddata/jdan/HDEEG/edf"
    params["dataFiles"][
        "ROOT"
    ] = "/users/sista/jdan/miniEEG"
    params["dataFiles"]["subject"] = subject
    params["dataFiles"]["edfFile"] = os.path.join(
        params["dataFiles"]["ROOT_EDF"],
        params["dataFiles"]["subject"] + ".edf"
    )
    params["dataFiles"]["eventFile"] = os.path.join(
        params["dataFiles"]["ROOT"], "Persyst",
        params["dataFiles"]["subject"] + ".xlsx"
    )
    params["dataFiles"]["locationFile"] = os.path.join(
        params["dataFiles"]["ROOT"],
        "electrodes.txt"
    )
    params['BASESAVE'] = "simulations/variableDistance/results"
    params['BASESAVE'] = os.path.join(params["dataFiles"]["ROOT"],
                                      params['BASESAVE'],
                                      params["dataFiles"]["subject"])



    results = dict()

    #################
    ### Load Data ###
    electrodes = candidates.loadElectrodeLocations(params["dataFiles"]["locationFile"])

    pairs, pairsI = candidates.getElecPairsWithDist(
        nb.typed.List(electrodes), params["distance"]
    )
    elecPool = candidates.selectNPairs(params["elecPoolSize"], pairs)

    # Load Data in MNE structure
    data = loading.loadHDData(
        params["dataFiles"]["edfFile"],
        params["dataFiles"]["eventFile"],
        params["dataFiles"]["locationFile"],
        params["fs"],
    )

    # Mne Data -> Numpy Data
    npData = data.get_data() * 1e6
    npData = loading.reRef(npData, pairsI[elecPool])

    # Remove non-physiological data
    power = spir.rolling_rms(npData, int(params["fs"] / 2))
    highPower = np.where(np.any(power > 100, axis=0))[0]
    for i in highPower:
        i0 = max(0, i-int(params["fs"] / 1))
        i1 = min(npData.shape[1], i+int(params["fs"] / 1))
        npData[:, i0:i1] = 0
    amplitude = np.abs(npData)
    highAmp = np.where(np.any(amplitude > 200, axis=0))[0]
    for i in highAmp:
        i0 = max(0, i-int(params["fs"] / 1))
        i1 = min(npData.shape[1], i+int(params["fs"] / 1))
        npData[:, i0:i1] = 0

    import gc
    del power, amplitude
    gc.collect()
    # Remove non-physiological annotations
    events = loading.loadEventFile(params["dataFiles"]["eventFile"], data.info["sfreq"])
    toKeep = []
    for i, event in enumerate(events):
        if np.any(npData[:, event[0]]):
            toKeep.append(i)

    mapping = {1: "IED"}
    annot_from_events = mne.annotations_from_events(
        events=events[toKeep],
        event_desc=mapping,
        sfreq=data.info["sfreq"],
        orig_time=data.info["meas_date"],
    )
    data.set_annotations(annot_from_events)

    # Downsample
    dataDown = scipy.signal.decimate(npData, int(params["fs"] / params["fsDown"]))

    results["pairs"] = pairs
    results["pairsI"] = pairsI
    results["elecPool"] = elecPool




    mapping = {'IED': 1}
    epochs = mne.Epochs(data,
                        events=events[toKeep],
                        tmin=-0.5,
                        tmax=0.5,
                        event_id=mapping,
                        preload=True)
    evoked = epochs['IED'].average()





    #######################
    ### Select Channels ###
    # Rss Down
    iedEvents = loading.mneAnnot2Events(data.annotations)

    rssDown = spir.build_cov_lw_norm(
        dataDown, iedEvents, params["lagDown"], params["fsDown"]
    )
    # Rnn Down
    bckgEvents = [(0, int(dataDown.shape[1] / params["fsDown"]))]

    rnnDown = spir.build_cov_lw(
        dataDown, bckgEvents, params["lagDown"], params["fsDown"]
    )

    # Channel Selection
    chSelection, objectives = forwardGreedySearchSVD(
        rssDown, rnnDown, params["numNodes"], params["lagDown"]
    )

    # results["rssDown"] = rssDown
    # results["rnnDown"] = rnnDown

    results["chSelection"] = chSelection
    results["objectives"] = objectives






    from mne.viz.topomap import _check_sphere, _pick_data_channels, pick_info, _find_topomap_coords
    pos = data.info
    sphere = _check_sphere(None, pos)
    picks = _pick_data_channels(pos, exclude=())  # pick only data channels
    pos = pick_info(pos, picks)
    picks = list(range(npData.shape[0]))
    pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)




    i = pairsI[elecPool][results["chSelection"][params["numNodes"]-1]].flatten()
    p = pairsI[elecPool][results["chSelection"][params["numNodes"]-1]]

    mask = np.zeros((len(evoked.data), len(evoked.data[0])))
    mask[i,:] = 1

    evoked.data = np.abs(evoked.data)
    fig = evoked.plot_topomap([0], time_unit='s', show=False, mask=mask,
                        mask_params=dict(markersize=10, markerfacecolor='y'))
    for a, b in p:
        fig.get_axes()[0].plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], 'y',
                               lw=3, zorder=1)

    fig.set_size_inches(16, 8)
    fig.get_axes()[0].set_title('')
    plt.ylim([0, np.max(evoked.data[:,50])*1e6])
    plt.gca().tick_params(labelsize=12)
    
    fig.add_subplot(1, 4, 4)
    evoked = epochs['IED'].average()
    PMax = np.max(np.abs(evoked.data.flatten()))
    for i in range(257):
        plt.plot(evoked.data[i]*1e6, 'k', alpha=(np.abs(evoked.data[i, 50])/PMax)**3)
    plt.ylabel('μV')
    plt.xlabel('times 1 sec')
    plt.xticks([])
    
    plt.savefig(subject + '.pdf', format='pdf')

    fig.show()
