# 3rd party lib
from numba import njit, prange
import numpy as np
import scipy
from scipy.signal import convolve
from sklearn.covariance import ledoit_wolf

# ## SP FUNCTIONS ###


def build_lagged_version(data, lags):
    lagged = np.zeros((data.shape[0] * lags, data.shape[1]), dtype=np.float64)
    for ch in range(len(data)):
        signal = data[ch]
        lagged[ch * lags : ch * lags + lags, :] = scipy.linalg.toeplitz(
            signal, np.zeros((lags, 1), dtype=np.float64)
        ).T
    return lagged


def build_cov(data, events, lags, fs):
    Rxx = np.zeros((data.shape[0] * lags, data.shape[0] * lags), dtype=np.float64)
    lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    for i, event in enumerate(events):
        i0 = int(event[0] * fs)
        i1 = int(event[1] * fs)

        lagged = build_lagged_version(data[:, i0:i1], lags)
        lagged -= lagged.mean(axis=1).reshape((lagged.shape[0]), 1)
        lagged_total = np.concatenate((lagged_total, lagged), axis=1)
        if lagged_total.shape[1] > 10000:
            cov = lagged_total @ lagged_total.T
            Rxx += (cov) / lagged_total.shape[1]
            lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    if lagged_total.shape[1]:
        cov = lagged_total @ lagged_total.T
        Rxx += (cov) / lagged_total.shape[1]

    return (Rxx + Rxx.T) / 2


def build_cov_lw(data, events, lags, fs):
    Rxx = np.zeros((data.shape[0] * lags, data.shape[0] * lags), dtype=np.float64)
    lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    totalCovSamples = 0
    
    for i, event in enumerate(events):
        i0 = int(event[0] * fs)
        i1 = int(event[1] * fs)
        
        lagged = build_lagged_version(data[:, i0:i1], lags)
        lagged -= lagged.mean(axis=1).reshape((lagged.shape[0]), 1)
        lagged_total = np.concatenate((lagged_total, lagged), axis=1)
        if lagged_total.shape[1] * lagged_total.shape[0] > 100e6:
            cov, shrink = ledoit_wolf(lagged_total.T)
            Rxx = Rxx * (
                totalCovSamples / (totalCovSamples + lagged_total.shape[1])
            ) + cov * (
                lagged_total.shape[1] / (totalCovSamples + lagged_total.shape[1])
            )
            totalCovSamples = totalCovSamples + lagged_total.shape[1]

            lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    if lagged_total.shape[1]:
        cov, shrink = ledoit_wolf(lagged_total.T)
        Rxx = Rxx * (
            totalCovSamples / (totalCovSamples + lagged_total.shape[1])
        ) + cov * (lagged_total.shape[1] / (totalCovSamples + lagged_total.shape[1]))

    return (Rxx + Rxx.T) / 2


def build_cov_lw_norm(data, events, lags, fs):
    Rxx = np.zeros((data.shape[0] * lags, data.shape[0] * lags), dtype=np.float64)
    lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    totalCovSamples = 0
    
    avgPower = []
    for i, event in enumerate(events):
        i0 = int(event[0] * fs)
        i1 = int(event[1] * fs)
        avgPower.append(np.sqrt(np.mean(data[:, i0:i1]**2)))
    avgPower = np.median(avgPower)
    for i, event in enumerate(events):
        i0 = int(event[0] * fs)
        i1 = int(event[1] * fs)
        
        eventPower = np.sqrt(np.mean(data[:, i0:i1]**2))
        if eventPower > avgPower:
            norm = avgPower / eventPower
        else:
            norm = 1

        lagged = build_lagged_version(data[:, i0:i1]*norm, lags)
        lagged -= lagged.mean(axis=1).reshape((lagged.shape[0]), 1)
        lagged_total = np.concatenate((lagged_total, lagged), axis=1)
        if lagged_total.shape[1] * lagged_total.shape[0] > 100e6:
            cov, shrink = ledoit_wolf(lagged_total.T)
            Rxx = Rxx * (
                totalCovSamples / (totalCovSamples + lagged_total.shape[1])
            ) + cov * (
                lagged_total.shape[1] / (totalCovSamples + lagged_total.shape[1])
            )
            totalCovSamples = totalCovSamples + lagged_total.shape[1]

            lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    if lagged_total.shape[1]:
        cov, shrink = ledoit_wolf(lagged_total.T)
        Rxx = Rxx * (
            totalCovSamples / (totalCovSamples + lagged_total.shape[1])
        ) + cov * (lagged_total.shape[1] / (totalCovSamples + lagged_total.shape[1]))

    return (Rxx + Rxx.T) / 2


def build_cov_events_lw(data, events, lags, fs):
    Rxx = np.zeros((data.shape[0] * lags, data.shape[0] * lags), dtype=np.float64)
    lagged_total = np.zeros((data.shape[0] * lags, 0), dtype=np.float64)
    for event in events:
        i0 = int(event[0] * fs)
        i1 = int(event[1] * fs)

        lagged = build_lagged_version(data[:, i0:i1], lags)
        lagged -= lagged.mean(axis=1).reshape((lagged.shape[0]), 1)
        lagged_total = np.concatenate((lagged_total, lagged), axis=1)
        Rxx += np.cov(lagged) / len(events)
    
    _, shrink = ledoit_wolf(lagged_total.T)
    mu = np.trace(Rxx) / np.shape(Rxx)[0]
    Rxx = (1 - shrink) * Rxx + shrink * mu * np.identity(np.shape(Rxx)[0])
    return (Rxx + Rxx.T) / 2


def build_cov_riemanian(data, events, lags, fs, weighted=False):
    Rxx = np.zeros((data.shape[0] * lags, data.shape[0] * lags), dtype=np.float64)
    totalCovSamples = 0
    if weighted:
        for event in events:
            i0 = int(event[0] * fs)
            i1 = int(event[1] * fs)
            totalCovSamples += i1 - i0
    for i in prange(len(events)):
        event = events[i]
        i0 = int(event[0] * fs)
        i1 = int(event[1] * fs)

        lagged = build_lagged_version(data[:, i0:i1], lags)
        lagged -= lagged.mean(axis=1).reshape((lagged.shape[0]), 1)

        cov, _ = ledoit_wolf(lagged.T)
        if weighted:
            Rxx += scipy.linalg.logm(cov) * ((i1 - i0) / totalCovSamples)
        else:
            Rxx += scipy.linalg.logm(cov)

    if not weighted:
        Rxx /= len(events)
    Rxx = np.real(scipy.linalg.expm(Rxx))

    return (Rxx + Rxx.T) / 2


def rolling_rms(data, duration):
    """Calculate rolling average RMS.

    Args:
        data: data contained in a vector
        duration: rolling average window duration in samples
    Return:
        power: rolling average RMS
    """
    power = np.square(data)
    if data.ndim == 2:
        power = np.apply_along_axis(lambda m: np.convolve(
            m, np.ones((duration,)), mode='same'), axis=1, arr=power)  / duration
    elif data.ndim == 1:
        power = convolve(power, np.ones((duration,)) / duration, mode="same")
    else:
        TypeError("Dimmension of data should be 1 or 2 to convolve.")
    power = np.abs(power)  # Fix machine precision issues
    return np.sqrt(power)


def covReref(Rxx, reref, lag):
    """Rereferencing of covariance matrix

    Args:
        Rxx
        reref: list of list of reref indices e.g. [[1, 0], [2, -3], [...]]
        lag: time lags per channel
    Return:
        RxxReref: Rxx reref
    """
    RxxReref = np.zeros((len(reref) * lag, len(reref) * lag))
    for x in range(len(reref)):
        for y in range(len(reref)):
            for i in range(len(reref[x])):
                for j in range(len(reref[y])):
                    for t1 in range(lag):
                        for t2 in range(lag):
                            if i:
                                signI = -1
                            else:
                                signI = 1
                            if j:
                                signJ = -1
                            else:
                                signJ = 1
                            RxxReref[x * lag + t1, y * lag + t2] = (
                                RxxReref[x * lag + t1, y * lag + t2]
                                + Rxx[
                                    (np.abs(reref[x][i])) * lag + t1,
                                    (np.abs(reref[y][j])) * lag + t2,
                                ]
                                * signI
                                * signJ
                            )
    return RxxReref


@njit(parallel=True)
def downsample(values, factor):
    """Downsample preserving spikes

    Args:
        values: ndArray (features x samples)
        factor: downsampling factor
    Return:
        downsampled_values: Downsampled ndArray (features x (samples/factor))
    """
    if values.ndim != 2:
        raise IndexError("Expecting a 2D ndArray (features x samples)")

    downsampled_values = np.zeros(
        (values.shape[0], int(np.round(values.shape[1] / factor)))
    )
    for i in prange(values.shape[0]):
        row = values[i]
        row = np.abs(row.copy())
        while row.any():
            j = np.argmax(row)
            u = min(int(np.round(j / factor)), downsampled_values.shape[1] - 1)
            if not downsampled_values[i, u]:
                downsampled_values[i, u] = row[j]
            row[
                max(int(j - np.ceil(factor / 2)), 0) : min(
                    int(j + np.ceil(factor / 2)), len(row)
                )
            ] = 0
    return downsampled_values


def correlationAvgEvents(data, events1, events2, duration, fs):
    avgEventsCh = []
    for i, events in enumerate([events1, events2]):
        avgEventsCh.append(np.zeros((len(events),len(data),int(duration*fs))))
        eventList = []
        for j, event in enumerate(events):
            i0 = int((event - duration/2)*fs)
            i1 = i0 + int(duration*fs)
            if i0 < 0 or i1 > data.shape[1]:
                continue
            imax = np.argmax(np.max(np.abs(data[:,i0:i1]), axis=0))-int(duration/2*fs)
            if i1+imax > data.shape[1]:
                continue
            avgEventsCh[i][j, :, :] = data[:, i0+imax:i1+imax]
            eventList.append([(i0+imax)/fs, (i1+imax)/fs])

    for i, events in enumerate([events1, events2]):
        avgEventsCh[i] = avgEventsCh[i].mean(axis=0)

    cor = list()
    weight = list()
    for i in range(len(data)):
        cor.append(np.corrcoef(avgEventsCh[0][i], avgEventsCh[1][i], rowvar=False)[0, 1])
        weight.append(np.sqrt(np.mean(avgEventsCh[1][i]**2)))
    return np.average(cor, weights=weight)

# ## SPIR ###


def calculate_filter(rss, rnn, regularization="none"):
    """Compute max-SNR filter

    Filter can be regularized to 95% Rss and 90% Rnn with PCA regularization

    Args:
        rss: data contained in a vector
        rnn: rolling average window duration in samples
        regularization: value 'none' or 'pca'
    Return:
        v: sorted eigenvectors of GEVD problem

    """
    ws, vs = np.linalg.eig(rss)
    index_s = np.argmax(np.cumsum(ws) / np.sum(ws) > 0.99)
    wn, vn = np.linalg.eig(rnn)
    index_n = np.argmax(np.cumsum(wn) / np.sum(wn) > 0.9)
    vt = np.concatenate((vn[:, : index_n + 1], vs[:, : index_s + 1]), axis=1)
    u, s, v = np.linalg.svd(vt, full_matrices=False)
    index_t = np.argmax(np.cumsum(s) / np.sum(s) > 0.999)
    i = index_t
    t = u

    if regularization == "none":
        w, v = np.linalg.eig(np.dot(np.linalg.inv(rnn), rss))
    elif regularization == "pca":
        w, v = np.linalg.eig(
            np.dot(
                np.dot(
                    np.dot(
                        t[:, 0:i],
                        np.linalg.inv(
                            np.dot(np.dot(np.transpose(t[:, 0:i]), rnn), t[:, 0:i])
                        ),
                    ),
                    np.transpose(t[:, 0:i]),
                ),
                rss,
            )
        )

    sortI = np.argsort(np.real(w))[::-1]

    return np.real(v[:, sortI]), np.real(w[sortI])


def maxspir_filter(data, v, noise):
    """Apply maxSPIR filter.

    Args:
        data: data contained in an array (row = channels, column = samples)
        v: maxSPIR filter as a flattened vector
        noise: noise binary mask contained in an array of the same size as data
    Return:
        out: filtered data
    """
    lag = int(len(v) / len(data))
    v_shaped = np.reshape(v, (len(data), lag))
    out = np.convolve(v_shaped[0, :], data[0, :] * np.logical_not(noise[0, :]), "same")
    for i in range(1, v_shaped.shape[0]):
        out += np.convolve(
            v_shaped[i, :], data[i, :] * np.logical_not(noise[i, :]), "same"
        )
    return out


# ## EVENT & MASK MANIPULATION ###


def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.

    Returns a logical array of length totalLen.
    All event epochs are set to True

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,), dtype=bool)
    for j, event in enumerate(events):
        i0 = min(int(np.round(event[0] * fs)), totalLen)
        i1 = min(int(np.round(event[1] * fs)), totalLen)
        if i1 == i0:
            i1 = min(i1 + 1, totalLen)
        for i in range(i0, i1):
            mask[i] = True
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.

    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]

    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask) - 1) / fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, end_i[0] / fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[start_i[-1] / fs, (len(mask) - 1) / fs]]
                start_i = np.delete(start_i, len(start_i) - 1)
        for i in range(len(start_i)):
            events.append([start_i[i] / fs, end_i[i] / fs])
        events += tmp
    return events


def extend_event(events, time, max_time):
    """Extends events in event list by time.

    The start time of each event is moved time seconds back and the end
    time is moved time seconds later

    Args:
        events: list of events. Each event is a tuple
        time: time to extend each event in seconds
        max_time: maximum end time allowed of an event.
    Return
        extended_events: list of events which each event extended.
    """
    extended_events = events.copy()
    for i, event in enumerate(events):
        extended_events[i] = [max(0, event[0] - time), min(max_time, event[1] + time)]
    return extended_events


def merge_events(events, distance):
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i - 1][1] < distance:
            events[i - 1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events


# ## SZ & INTERFERENCE DETECTION ###


def find_interference(data, fs, seizures, maxThresh=500, minTresh=70):
    """Find interference in raw data.
    Args:
        data: raw data contained in an array (row = channels, column = samples)
        fs: sampling frequency in Hz
        duration: duration of each event in seconds [default:30]
        minTreshold: minimum detection threshold in mV [default:70]
    Return:
        interferences: indices of detected events (sorted with decreasing power)
    """
    threshold = 9999
    power = rolling_rms(data, int(1 * fs))

    dpower = np.diff(power)
    s_len = int(fs / 2)
    dpower = convolve(dpower, np.ones((dpower.shape[0], s_len)) / s_len, "same")
    seizure_mask = eventList2Mask(seizures, power.shape[1], fs)

    interference_mask = np.zeros((data.shape[1],))

    for c, channel in enumerate(power):
        np.putmask(channel, seizure_mask, 0)
        events = list()
        while threshold > minTresh and len(events) < 50:
            i = np.argmax(channel)
            threshold = channel[i]

            # Start
            i0 = i - s_len
            while i0 > 0 and dpower[c, i0] > 0:
                i0 -= 1
            i0 += s_len
            # End
            i1 = i + s_len
            while i1 < dpower.shape[1] and dpower[c, i1] < 0:
                i1 += 1
            i1 -= s_len

            np.put(
                channel, np.arange(max(0, i0 - s_len), min(len(channel), i1 + s_len)), 0
            )
            if (
                threshold > minTresh
                and threshold < maxThresh
                and i1 - i0 < 60 * fs
                and i1 - i0 > fs
            ):
                events.append((i0 / fs, i1 / fs))
        eventmask = eventList2Mask(events, len(interference_mask), fs)
        interference_mask = np.logical_or(interference_mask, eventmask)

    return mask2eventList(interference_mask, fs)


def find_events(power, data, fs, duration=30, minTresh=100, minNoise=500):
    """Find events in power of filtered data.

    Args:
        power: power of filtered data (single channel)
        data: raw data contained in an array (row = channels, column = samples)
        fs: sampling frequency in Hz
        duration: duration of each event in seconds [default:30]
        minTreshold: minimum detection threshold in mV [default:100]
        minNoise: minimum noise threshold in mv [default: 500mV]
    Return:
        events: indices of detected events (sorted with decreasing power)
        noise_events: indices of detected noise events
    """
    threshold = 9999
    power_copy = np.copy(power)
    events, noise_events = list(), list()
    iteration = 0
    while threshold > minTresh and iteration < 1000:
        iteration += 1
        i = np.argmax(power_copy)
        threshold = power_copy[i]
        i0 = max(0, int(i - duration / 2 * fs))
        i1 = min(int(i + duration / 2 * fs), len(power_copy))
        if np.median(np.max(np.abs(data[:, i0:i1]), axis=1)) > minNoise:
            noise_events.append(i / fs)
        else:
            events.append(i / fs)
        np.put(power_copy, np.arange(i0, i1), 0)
    return events[:-1], noise_events


def seizure_power(power, seizures, fs):
    """Return max power during seizure event

    Args:
        power: power of filtered data (single channel)
        seizures: list of tupes containing seizure start and end times in sec
        fs: sampling frequency in Hz
    Return:
        sz_power: list of max power during seizure event
    """

    sz_power = list()
    for event in seizures:
        i0 = int(fs * event[0])
        i1 = int(fs * event[1])
        sz_power.append(np.max(power[i0:i1]))
    return sz_power


def score_events(seizures, events, duration, ictalEvents=None):
    """Compute TP and FP count given detected events and reference annotations

    Detections during ictal events are neither considered as TP nor as FP.

    Args:
        seizures: reference list of tupes containing seizure start and end times in sec
        events: detected list of events times in sec
        duration: sec offset between detected event and true event
        ictalEvents: reference list of tupes containing seizure start and end times in sec

    Return:
        tp: binary list containing 1 for detected seizures and 0 for missed ones
        fp: list of FP events
    """
    if ictalEvents is None:
        ictalEvents = seizures
    tp = np.zeros(
        len(seizures),
    )
    fp = list()
    for event in events:
        found = False
        for i, seizure in enumerate(seizures):
            if seizure[0] - duration < event and event < seizure[1] + duration:
                found = True
                tp[i] = 1
                break
        if not found:
            for i, seizure in enumerate(ictalEvents):
                if seizure[0] - duration < event and event < seizure[1] + duration:
                    found = True
                    break
            if not found:
                fp.append(event)
    return tp, fp
