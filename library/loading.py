import mne
import numpy as np
from numba import njit
import pandas as pd


def loadHDData(edfFile, eventFile, montageFile, fs=100):
    # Load Data in MNE structure
    data = mne.io.read_raw_edf(edfFile, exclude=['ECG'], preload=True)

    digiMontage = mne.channels.read_custom_montage(montageFile)
    data.set_montage(digiMontage)

    data = data.set_eeg_reference(ref_channels="average")

    data.filter(l_freq=3, h_freq=30)
    data = data.resample(sfreq=fs)

    # Events, annotations and epochs
    events = loadEventFile(eventFile, data.info["sfreq"])

    mapping = {1: "IED"}
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=mapping,
        sfreq=data.info["sfreq"],
        orig_time=data.info["meas_date"],
    )
    data.set_annotations(annot_from_events)

    return data


def loadEventFile(eventFile, fs, confidence=0.9):
    events = list()
    df = pd.read_excel(eventFile)
    x = df[df.Amplitude < 150]
    x = x[x.Perception > confidence]
    # x = x.groupby('Channel').filter(lambda grp: grp.Channel.count() > 5)
    for event in list(x.Time):
        events.append([int(event * fs), 0, 1])
    events = np.array(events)

    return events


@njit
def reRef(data, pairs):
    shortData = np.empty((len(pairs), data.shape[1]))
    for i, pair in enumerate(pairs):
        shortData[i] = data[pair[1]] - data[pair[0]]
    return shortData


def mneAnnot2Events(annotations, t0=-0.2, t1=0.2):
    events = list()
    for event in annotations:
        i0 = event["onset"]
        events.append([i0 + t0, i0 + t1])
    return events
