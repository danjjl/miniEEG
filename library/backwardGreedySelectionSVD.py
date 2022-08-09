import numpy as np
from numba import njit, prange


def backwardGreedySearchSVD(R1, R2, nbGroupToSel, lag):
    """
    BACKWARDGREEDYSEARCH Perform a backward group selection search for the
    GEVD-problem of the matrix pencil (R1,R2).

    Input parameters:
      R1 [DOUBLE]: the target covariance matrix
      R2 [DOUBLE]: the interference covariance matrix
      nbGroupToSel [INTEGER]: the number of groups to select
      groupSelector [BINARY]: a nbVariables.nbGroups x nbGroups binary
          matrix, indicating per group (column) which variables of the
          covariance matrices belong to that group with ones at the
          corresponding positions.
      K [INTEGER]: the number of channels to select

    Output parameters:
      groupSel [INTEGER]: the groups that are selected
      maxObjFun [DOUBLE]: the corresponding objective (i.e., generalized
          Rayleigh quotient)
    """
    nbGroup = int(len(R1) / lag)
    chSelector = np.zeros((len(R1), nbGroup))
    for ch in range(nbGroup):
        chSelector[ch * lag : ch * lag + lag, ch] = 1
    groupSelector = chSelector

    selected = list()
    objective = list()

    selectedGroups = np.ones((nbGroup, 1))
    for _ in range(nbGroup - nbGroupToSel):
        objFuns = np.ones((nbGroup, 1)) * -9999
        for j in prange(nbGroup):
            if selectedGroups[j]:
                selectedGroups[j] = 0
                sel = np.array(
                    np.sum(groupSelector[:, np.where(selectedGroups)[0]], 1), dtype=int
                )
                rss = R1[np.where(sel)][:, np.where(sel)].reshape(
                    np.sum(sel), np.sum(sel)
                )
                rnn = R2[np.where(sel)][:, np.where(sel)].reshape(
                    np.sum(sel), np.sum(sel)
                )
                selectedGroups[j] = 1
                objFuns[j] = largestNormlizedEigValGEVD(rss, rnn)
        optCombI = np.argmax(objFuns)
        selectedGroups[optCombI] = 0

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
