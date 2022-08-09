import numpy as np
from numba import njit, prange


def forwardGreedySearchSVD(R1, R2, nbGroupToSel, lag):
    nbGroup = int(len(R1) / lag)
    chSelector = np.zeros((len(R1), nbGroup))
    for ch in range(nbGroup):
        chSelector[ch * lag : ch * lag + lag, ch] = 1
    groupSelector = chSelector

    selected = list()
    objective = list()

    selectedGroups = np.zeros((nbGroup, 1))
    for _ in range(nbGroupToSel):
        objFuns = np.ones((nbGroup, 1)) * -9999
        for j in prange(nbGroup):
            if not selectedGroups[j]:
                selectedGroups[j] = 1
                sel = np.array(
                    np.sum(groupSelector[:, np.where(selectedGroups)[0]], 1), dtype=int
                )
                rss = R1[np.where(sel)][:, np.where(sel)].reshape(
                    np.sum(sel), np.sum(sel)
                )
                rnn = R2[np.where(sel)][:, np.where(sel)].reshape(
                    np.sum(sel), np.sum(sel)
                )
                selectedGroups[j] = 0
                objFuns[j] = largestNormlizedEigValGEVD(rss, rnn)
        optCombI = np.argmax(objFuns)
        selectedGroups[optCombI] = 1

        selected.append(np.where(selectedGroups)[0])
        objective.append(objFuns[optCombI])

    return selected, objective


def largestNormlizedEigValGEVD(rss, rnn):
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

    w = np.linalg.eigvals(rnnCInv @ rss)

    return np.max(np.real(w))