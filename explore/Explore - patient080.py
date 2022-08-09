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


subject = 'patient080'
distance = 5.0
numNodes = 10

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

# +
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

# +
plt.figure(figsize=(16, 6))
plt.plot([x+1 for x in range(len(results["objectives"]))], results["objectives"], '.')
plt.ylabel('GEVD objective')
plt.xlabel('number of nodes')
plt.title('Channel selection objective in function of number of nodes')
plt.show()


display(Markdown('### Selected Channels'))
print(results["chSelection"][params["numNodes"]-1])
fig = plot.plotElectrodePairs(pairs[elecPool][results["chSelection"][params["numNodes"]-1]])
plt.title('Selected Electrodes')
plt.show()


display(Markdown('### 15 Channels (including selected)'))
rms = np.sqrt(np.mean(dataDown**2, axis=1))
chToPlot = list(np.argsort(rms)[::-1][:15])
j = len(chToPlot) - 1
colored = list()
for selected in results["chSelection"][params["numNodes"]-1]:
    if selected not in chToPlot:
        while chToPlot[j] in results["chSelection"][params["numNodes"]-1]:
            j-= 1
        chToPlot[j] = selected
        colored.append(j)
        j -= 1
    else:
        colored.append(chToPlot.index(selected))
plot.plotEvents(npData[chToPlot], params["fs"], iedEvents, colored=colored)

display(Markdown('### Downsampled selected channels'))
plot.plotEvents(dataDown[results["chSelection"][params["numNodes"]-1]], params["fsDown"], iedEvents)

display(Markdown('### Downsampled average IED'))
plot.plotAvgIed(dataDown[results["chSelection"][params["numNodes"]-1]]*10, params["fsDown"], iedEvents)



covPlots = [
    {
        'title': 'Rss',
        'data': rssDown},
    {
        'title': 'Rnn',
        'data': rnnDown}]
plt.figure(figsize=(16, 9))
for i, plotItem in enumerate(covPlots):
    plt.subplot(1, len(covPlots), i + 1)
    plt.title(plotItem['title'])
    plt.imshow(plotItem['data'])
    plt.xticks(np.arange(0, params["lagDown"] * params["elecPoolSize"], step=params["lagDown"]))
    plt.yticks(np.arange(0, params["lagDown"] * params["elecPoolSize"], step=params["lagDown"]))

plt.tight_layout()
plt.show()

# +
##############################
### Compute max-SNR filter ###
selectedData = npData[results["chSelection"][params["numNodes"] - 1]]
rss = spir.build_cov_lw_norm(selectedData, iedEvents, params["lag"], params["fs"])
rnn = spir.build_cov_lw(selectedData, bckgEvents, params["lag"], params["fs"])

ws, vs = np.linalg.eigh(rss)
ws, vs = np.real(ws[::-1]), np.real(vs[:, ::-1])
index_s = np.argmax(np.cumsum(ws) / np.sum(ws) > 0.95)

wn, vn = np.linalg.eigh(rnn)
wn, vn = np.real(wn[::-1]), np.real(vn[:, ::-1])
index_n = np.argmax(np.cumsum(wn) / np.sum(wn) > 0.85)

vt = np.concatenate((vn[:, : index_n + 1], vs[:, : index_s + 1]), axis=1)
u, s, _ = np.linalg.svd(vt, full_matrices=False)
index_t = np.argmax(np.cumsum(s) / np.sum(s) > 0.99)
compression = np.real(u[:, :index_t])

rnnC = compression.T @ rnn @ compression
rnnC = (rnnC + rnnC.T) / 2
rnnCInv = np.linalg.inv(rnnC)
rnnCInv = np.real(rnnCInv)
rnnCInv = (rnnCInv + rnnCInv.T) / 2
rnnCInv = compression @ rnnCInv @ compression.T
rnnCInv = (rnnCInv + rnnCInv.T) / 2 + np.eye(rnnCInv.shape[0])*1e-15

w, v = np.linalg.eig(rnnCInv @ rss)
i = np.argsort(np.real(w))[::-1]
w, v = np.real(w[i]), np.real(v[:, i])

lagged = spir.build_lagged_version(selectedData, params["lag"])

nOutTime = (v[:, :1].T @ lagged)
power = spir.rolling_rms(nOutTime[:1, :], params["lag"])

# results["rss"] = rss
# results["rnn"] = rnn
results["w"] = w
results["v"] = v

# +
# %matplotlib inline
plt.figure(figsize=(16, 6))
plt.plot(results['w'], '.')
plt.title('Eigenvalues of Rss @ Rnn$^{-1}$')
plt.show()

norm = np.sqrt(np.mean(power**2))/np.sqrt(np.mean(selectedData**2))

plot.plotEvents(np.concatenate((power/norm, selectedData)), params["fs"], iedEvents,
    ["max-SNR"] + ["EEG {}".format(j) for j in range(selectedData.shape[0])],[0])
plot.plotAvgIed(np.concatenate((power/norm, selectedData)), params["fs"], iedEvents)

plt.figure(figsize=(2, 6))
plt.title('Filter')
for i in range(params["numNodes"]):
    plt.plot(v[i*params['lag']:(i+1)*params['lag'], 0] -0.3*i, 'k')
plt.tick_params(labelleft=False, left=False)
plt.show()

covPlots = [
    {
        'title': 'Rss',
        'data': rss},
    {
        'title': 'Rnn',
        'data': rnn}]
plt.figure(figsize=(16, 9))
for i, plotItem in enumerate(covPlots):
    plt.subplot(1, len(covPlots), i + 1)
    plt.title(plotItem['title'])
    plt.imshow(plotItem['data'])
    plt.xticks(np.arange(0, params["lag"] * params["numNodes"], step=params["lag"]))
    plt.yticks(np.arange(0, params["lag"] * params["numNodes"], step=params["lag"]))

plt.tight_layout()
plt.show()

# +
################
### Classify ###
signalPower = spir.rolling_rms(selectedData, int(params["fs"]))
signalDownPower = np.mean(signalPower, axis=0)

XsUp = np.concatenate((power, signalDownPower.reshape(1, len(signalDownPower))))
Xs = spir.downsample(XsUp, int(params["fs"] / params["fsClassify"]))
Xs = Xs.T

ys = ied.IedDetector.buildLabels(
    [x["onset"] for x in data.annotations], Xs[:, 0], params["fsClassify"]
)

iedDetector = ied.IedDetector()

# Keep only region of intest
indices = iedDetector.findRegionOfInterest(Xs, ys)
XTrain = Xs[indices, :]
yTrain = ys[indices]

# Fit an LDA
iedDetector.fit(XTrain, yTrain)

# Test
predictions_proba = iedDetector.predict(Xs)
auc = roc_auc_score(ys, predictions_proba)
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    threshold = iedDetector.findThreshold(predictions_proba, ys, t)
    predictions = (predictions_proba >= threshold).astype(bool)
    fp = np.sum(np.logical_and(predictions, ys == 0))
    results["threshold-{}".format(t)] = threshold
    results["fp-{}".format(t)] = fp
    results["f1-{}".format(t)] = f1_score(ys, predictions)
    results["cohenKappa-{}".format(t)] = cohen_kappa_score(ys, predictions)
    events1 = list(np.array(np.where(predictions)[0])/params['fsClassify'])
    events2 = list(np.array(np.where(ys)[0])/params['fsClassify'])
    results["cor-{}".format(t)] = spir.correlationAvgEvents(selectedData, events1, events2, 0.25, params['fs'])
results["auc"] = auc
results["ied"] = iedDetector


## Compute SNR
iedMask = spir.eventList2Mask(
    iedEvents, selectedData.shape[1], params["fs"]
)

bckgMask = spir.eventList2Mask(
    bckgEvents, selectedData.shape[1], params["fs"]
)

s = np.mean(signalPower[:, iedMask])
n = np.mean(signalPower[:, bckgMask])
results["signal"] = s
results["noise"] = n
results["snr"] = s / n

sMaxSnr = np.mean(power[:, iedMask].flatten())
nMaxSnr = np.mean(power[:, bckgMask].flatten())
results["signalMaxSnr"] = sMaxSnr
results["noiseMaxSnr"] = nMaxSnr
results["snrMaxSnr"] = sMaxSnr / nMaxSnr

# +
iedDetector.plotFeatures(Xs, ys, ['Power max-SNR', 'mean signal power'],
                         threshold)

print('# Confusion matrix')
confMat = confusion_matrix(ys, predictions)
display(pd.DataFrame(confMat))
ied.plotROC(predictions_proba, ys, confMat, sensitivity=0.2)

display(Markdown('### IED'))
spikeEvents = np.where(ys)[0]/params["fsClassify"]
for i in range(min(int(np.ceil(len(data.annotations)/25)), 4)):
    ied.plotEvents(selectedData.T,
       XsUp.T/norm, predictions, data.info['sfreq'],
       params["fsClassify"], spikeEvents[i*25:min(len(spikeEvents), (i+1)*25)])

display(Markdown('### FP'))
interferenceEvents = np.where(np.logical_and((predictions), ys==0))[0]/params["fsClassify"]
for i in range(min(int(len(interferenceEvents)/25), 4)):
    ied.plotEvents(selectedData.T,
               XsUp.T/norm, 1-predictions, data.info['sfreq'],
               params["fsClassify"], interferenceEvents[i*25:min(len(interferenceEvents), (i+1)*25)])

# +
duration = 0.25
fs = params['fs']

avgEvents = []
for i, events in enumerate([events1, events2]):
    eventList = []
    avgEvents.append([])
    for event in events:
        i0 = int((event - duration/2)*fs)
        i1 = i0 + int(duration*fs)
        if i0 < 0 or i1 > selectedData.shape[1]:
            continue
        imax = np.argmax(np.max(np.abs(selectedData[:,i0:i1]), axis=0))-int(duration/2*fs)
        if i1+imax > selectedData.shape[1]:
            continue
        avgEvents[i].append(selectedData[:, i0+imax:i1+imax].flatten())
        eventList.append([(i0+imax)/fs, (i1+imax)/fs])
    avgEvents[i] = np.mean(avgEvents[i], axis=0)
    plot.plotAvgIed(selectedData*4, params["fs"], eventList)
print(np.corrcoef(avgEvents[0], avgEvents[1], rowvar=False)[0, 1])
# -

for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    print(results["cor-{}".format(t)])


