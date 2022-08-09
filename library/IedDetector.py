import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import PowerTransformer


class IedDetector:
    clf = None
    logT = None
    ranges = None

    def __init__(self):
        self.clf = LinearDiscriminantAnalysis()

    def buildLabels(events, x, fs):
        """Build label vector from events

        Allows one sample offset to match peak of feature 0

        Args:
            events : list of events in seconds
            x : 1D ndArray data array
            fs : sampling frequency
        Return:
            ys: 1D ndArray containing ones when event is positive
        """
        ys = np.zeros((x.shape[0],))
        for i in events:
            i = int(np.round(i * fs))
            ys[i] = 1
        for i in np.where(ys == 1)[0][0::-1]:
            if i + 1 == x.shape[0]:
                continue
            if x[i] < x[i + 1] and not ys[i + 1]:
                ys[i] = 0
                ys[i + 1] = 1
        return ys

    def predict(self, X):
        XLog = self.logT.transform(np.abs(X) + 1e-24)
        predictions_proba = self.clf.predict_proba(XLog)[:, 1]

        for i in range(X.shape[1]):
            if self.ranges is not None:
                predictions_proba[X[:, i] < self.ranges[0, i]] = 0
                predictions_proba[X[:, i] > self.ranges[1, i]] = 0

        return predictions_proba

    def findRegionOfInterest(self, X, y, margin=0.2):
        "Find min and max values of each feature for positive labels"

        indices = np.ones((X.shape[0]))
        self.ranges = np.zeros((2, X.shape[1]))
        high = 1 + margin
        low = 1 - margin
        for i in range(X.shape[1]):
            minRange = np.min(X[:, i][np.where(y == 1)[0]])
            minRange = high * minRange if minRange < 0 else low * minRange
            maxRange = np.max(X[:, i][np.where(y == 1)[0]])
            maxRange = low * maxRange if maxRange < 0 else high * maxRange
            indices = np.logical_and(
                indices, np.logical_and(minRange < X[:, i], X[:, i] < maxRange)
            )
            self.ranges[0, i] = minRange
            self.ranges[1, i] = maxRange
        return indices

    def findThreshold(self, predictions, y, sensitivity=0.9):
        return np.percentile(predictions[np.where(y == 1)[0]], 100 - 100 * sensitivity)

    def fit(self, X, y):
        self.logT = PowerTransformer(method="box-cox").fit(np.abs(X) + 1e-24)
        XLog = self.logT.transform(np.abs(X) + 1e-24)
        self.clf = self.clf.fit(XLog, y)

    def plotFeatures(self, X, y, labels=None, threshold=None):
        # Plot Classification
        plt.figure(figsize=(16, 6))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        t = 0.05

        if threshold is not None:
            h = 200

            xx, yy = np.meshgrid(
                np.linspace(
                    np.percentile(X[:, 0], t), np.percentile(X[:, 0], 100 - t), h
                ),
                np.linspace(
                    np.percentile(X[:, 1], t), np.percentile(X[:, 1], 100 - t), h
                ),
            )

            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
            plt.contour(xx, yy, Z, [threshold], linewidths=2.0, colors="white")

        plt.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=cm_bright,
            edgecolors="k",
            alpha=0.6,
        )
        plt.gca().set_xlim(np.percentile(X[:, 0], t), np.percentile(X[:, 0], 100 - t))
        plt.gca().set_ylim(np.percentile(X[:, 1], t), np.percentile(X[:, 1], 100 - t))
        if labels is not None:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])

        plt.show()


def plotROC(predictions, y, confMat, sensitivity=0.9):
    fpr, tpr, _ = roc_curve(y, predictions)
    fIMax = np.where(tpr == 1)[0][0]

    plt.figure(figsize=(16, 4))
    lw = 2
    plt.plot(
        [0, confMat[0, 1]], [sensitivity, sensitivity], "--", color="navy", alpha=0.5
    )
    plt.plot(
        [confMat[0, 1], confMat[0, 1]], [0, sensitivity], "--", color="navy", alpha=0.5
    )
    plt.scatter(confMat[0, 1], sensitivity, marker="o", s=50, color="navy")
    plt.plot(
        fpr * np.sum(y == 0),
        tpr,
        "-",
        color="darkorange",
        lw=lw,
        label="ROC curve (area = {:.4f})".format(roc_auc_score(y, predictions)),
    )
    text = """\
    Sensitivity    : {:.2f}
    False alarms : {}
    """
    plt.annotate(
        text.format(sensitivity, confMat[0, 1]),
        color="navy",
        xy=(confMat[0, 1], sensitivity),
        xytext=(confMat[0, 1] * 1.1, 0.6),
        arrowprops=dict(arrowstyle="->", connectionstyle="angle3", color="navy"),
    )
    plt.xlim([0.0, (fpr[fIMax] + 0.1 * fpr[fIMax]) * np.sum(y == 0)])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positives")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


def plotAvgIed(data, X, fs, events):
    # Average IED
    Xavg = list()
    for j in range(X.shape[1]):
        Xavg.append(list())
    dataAvg = list()
    for j in range(data.shape[1]):
        dataAvg.append(list())

    for i in range(len(events)):
        t0 = int((events[i] - 0.5) * fs)
        t1 = t0 + int(fs)
        for j in range(X.shape[1]):
            Xavg[j].append(X[t0:t1, j] - np.mean(X[t0:t1, j]))
        for j in range(data.shape[1]):
            dataAvg[j].append(data[t0:t1, j] - np.mean(data[t0:t1, j]))

    Xavg = np.mean(Xavg, axis=1)
    dataAvg = np.mean(dataAvg, axis=1)

    plt.figure(figsize=(3, 3))
    plt.title("Avg IED")

    t = np.arange(fs) / fs

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for j in range(X.shape[1]):
        plt.plot(t, Xavg[j] - 100 * (j), colors[j])
    for j in range(data.shape[1]):
        plt.plot(t, dataAvg[j] - 100 * (j + X.shape[1]), "k")

    plt.yticks(
        [-100 * (j) for j in range(data.shape[1] + X.shape[1])],
        ["Feat {}".format(j) for j in range(X.shape[1])]
        + ["EEG {}".format(j) for j in range(data.shape[1])],
    )
    plt.xlim(0, 1)
    plt.xlabel("Time [s]")
    plt.show()


def plotEvents(data, X, predictions, fs, fsDown, events):
    # Plot events

    plt.figure(figsize=(16, 6))
    nEvents = min(25, len(events))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    for i in range(nEvents):
        t0 = int((events[i] - 0.5) * fs)
        t1 = int(t0 + fs)
        if t0 < 0 or t1 > X.shape[0]:
            continue
        t = (np.arange(t0, t1) - t0) / fs + i
        tD = int(np.round(events[i] * fsDown))

        if predictions[tD]:
            plt.axvspan(i, i + 1, facecolor="g", alpha=0.1)
        else:
            plt.axvspan(i, i + 1, facecolor="r", alpha=0.1)
        plt.axvline(i, color="gray")

        for j in range(X.shape[1]):
            d = X[t0:t1, j]
            m = np.mean(d)
            plt.plot(t, X[t0:t1, j] - 100 * (j), color=colors[j])
        for j in range(data.shape[1]):
            d = data[t0:t1, j]
            m = np.mean(d)
            plt.plot(t, (d - m) - 100 * (j + X.shape[1]), "k")

    plt.yticks(
        [-100 * (j) for j in range(data.shape[1] + X.shape[1])],
        ["Feat {}".format(j) for j in range(X.shape[1])]
        + ["EEG {}".format(j) for j in range(data.shape[1])],
    )
    plt.xticks([0.5 + i for i in range(nEvents)], [i for i in range(nEvents)])
    plt.xlim(0, nEvents)
    plt.show()


def plotOverview(data, X, XDown, y, predictions, fs, fsDown):
    # Classification overview
    t = np.arange(data.shape[0]) / fs
    tDown = np.arange(XDown.shape[0]) / fsDown

    plt.figure(figsize=(16, 6))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    fp = np.where(np.logical_and(predictions, y == 0))[0]
    for event in fp:
        plt.axvline(event / fsDown, color="orange", alpha=0.5)
    fn = np.where(np.logical_and(~predictions, y))[0]
    for event in fn:
        plt.axvline(event / fsDown, color="blue", alpha=0.5)
    tp = np.where(np.logical_and(predictions, y))[0]
    for event in tp:
        plt.axvline(event / fsDown, color="green", alpha=0.5)

    for j in range(X.shape[1]):
        plt.plot(t, X[:, j] - 100 * (j), color=colors[j], alpha=0.7)
        plt.plot(tDown, XDown[:, j] - 100 * (j), ".", color=colors[j])
    for j in range(data.shape[1]):
        plt.plot(t, data[:, j] - 100 * (j + X.shape[1]), "k", alpha=0.7)

    plt.xlabel("Time [s]")
    plt.yticks(
        [-100 * (j) for j in range(data.shape[1] + X.shape[1])],
        ["Feat {}".format(j) for j in range(X.shape[1])]
        + ["EEG {}".format(j) for j in range(data.shape[1])],
    )
    plt.xlim(0, data.shape[0] / fs)
    plt.ylim(-100 * (data.shape[1] + X.shape[1]) - 100, 200)

    plt.show()
