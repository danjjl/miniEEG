import os
import pickle
import sys

import mne
import numba as nb
import numpy as np
import scipy.signal
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

sys.path.append("library")
import candidates
from forwardGreedySelection import forwardGreedySearchSVD
import IedDetector as ied
import loading
import spir


def crossValidate(params):

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

    #######################
    ### Select Channels ###
    # Rss Down
    annotSel = np.ones((len(data.annotations),), dtype=bool)
    chunk = len(annotSel) / params["NFOLDS"]
    annotSel[
        int(params["testFold"] * chunk) : int((params["testFold"] + 1) * chunk)
    ] = 0
    iedEvents = loading.mneAnnot2Events(data.annotations[annotSel])

    rssDown = spir.build_cov_lw_norm(
        dataDown, iedEvents, params["lagDown"], params["fsDown"]
    )
    # Rnn Down
    bckgEvents = list()
    chunk = dataDown.shape[1] / params["fsDown"] / params["NFOLDS"]
    if params["testFold"] > 0:
        bckgEvents.append((0, chunk * params["testFold"]))
    if params["testFold"] < params["NFOLDS"] - 1:
        bckgEvents.append(
            (
                (params["testFold"] + 1) * chunk,
                int(dataDown.shape[1] / params["fsDown"]),
            )
        )

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

    for params["numNodes"] in range(1, 11):
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

        nOutTime = (v[:, :3].T @ lagged)
        power = spir.rolling_rms(nOutTime[:3, :], params["lag"])

        # results["rss"] = rss
        # results["rnn"] = rnn
        results["w"] = w
        results["v"] = v

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

        selector = np.ones((len(ys),), dtype=bool)
        chunk = len(selector) / params["NFOLDS"]
        selector[
            int(params["testFold"] * chunk) : int((params["testFold"] + 1) * chunk)
        ] = 0
        ysSelect = ied.IedDetector.buildLabels(
            [x["onset"] for x in data.annotations[annotSel]],
            Xs[:, 0],
            params["fsClassify"],
        )
        # Add all IED
        selector = np.logical_or(selector, ys)
        # Remove test IED
        selector[np.logical_xor(ys, ysSelect)] = 0
        XTrain = Xs[selector]
        yTrain = ys[selector]

        # Keep only region of intest
        indices = iedDetector.findRegionOfInterest(XTrain, yTrain)
        XTrain = XTrain[indices, :]
        yTrain = yTrain[indices]
        iedDetector.findRegionOfInterest(Xs, ys)

        # Fit an LDA
        iedDetector.fit(XTrain, yTrain)

        # Test
        XTest = Xs[~selector]
        yTest = ys[~selector]
        predictions_proba = iedDetector.predict(XTest)
        predictions_probaAll = iedDetector.predict(Xs)
        fs10 = (1.0/10)
        x10 = spir.downsample(predictions_proba.reshape(1, len(predictions_proba)), int(params["fsClassify"] / fs10))[0]
        y10 = spir.downsample(yTest.reshape(1, len(yTest)), int(params["fsClassify"] / fs10))[0]
        auc = roc_auc_score(yTest, predictions_proba)
        auc10 = roc_auc_score(y10, x10)
        for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
            threshold = iedDetector.findThreshold(predictions_proba, yTest, t)
            predictions = (predictions_proba >= threshold).astype(bool)
            fp = np.sum(np.logical_and(predictions, yTest == 0))
            results["threshold-{}".format(t)] = threshold
            results["fp-{}".format(t)] = fp
            results["f1-{}".format(t)] = f1_score(yTest, predictions)
            results["cohenKappa-{}".format(t)] = cohen_kappa_score(yTest, predictions)
            predictionsAll = (predictions_probaAll >= threshold).astype(bool)
            events1 = list(np.array(np.where(~selector*predictionsAll)[0])/params['fsClassify'])
            events2 = list(np.array(np.where(~selector*ys)[0])/params['fsClassify'])
            results["cor-{}".format(t)] = spir.correlationAvgEvents(selectedData, events1, events2, 0.35, params['fs'])
            
            t10 = iedDetector.findThreshold(x10, y10, t)
            pred10 = (x10 >= t10).astype(bool)
            results["cohenKappa10-{}".format(t)] = cohen_kappa_score(y10, pred10)
        results["auc"] = auc
        results["auc10"] = auc10
        results["ied"] = iedDetector


        

        ## Compute SNR
        testIedEvents = loading.mneAnnot2Events(data.annotations[~annotSel])
        testIedMask = spir.eventList2Mask(
            testIedEvents, selectedData.shape[1], params["fs"]
        )
        testBckgEvents = list()
        chunk = dataDown.shape[1] / params["fsDown"] / params["NFOLDS"]
        testBckgEvents.append(
            (chunk * params["testFold"], (params["testFold"] + 1) * chunk)
        )
        testBckgMask = spir.eventList2Mask(
            testBckgEvents, selectedData.shape[1], params["fs"]
        )

        s = np.mean(signalPower[:, testIedMask])
        n = np.mean(signalPower[:, testBckgMask])
        results["signal"] = s
        results["noise"] = n
        results["snr"] = s / n

        sMaxSnr = np.mean(power[:, testIedMask].flatten())
        nMaxSnr = np.mean(power[:, testBckgMask].flatten())
        results["signalMaxSnr"] = sMaxSnr
        results["noiseMaxSnr"] = nMaxSnr
        results["snrMaxSnr"] = sMaxSnr / nMaxSnr

        simulation = dict()
        simulation["results"] = results
        simulation["parameters"] = params

        with open(
            os.path.join(
                params["BASESAVE"],
                str(params["numNodes"]),
                "simulation-d{}-f{}.pkl".format(params["distance"], params["testFold"]),
            ),
            "wb",
        ) as file:
            pickle.dump(simulation, file)
