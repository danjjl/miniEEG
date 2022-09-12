
import os
import sys

from crossvalidateFw import crossValidate

if __name__ == '__main__':
    subject = sys.argv[1]
    distance = float(sys.argv[2])
    fold = int(sys.argv[3])

    params = dict()
    params["IEDDURATION"] = 0.2

    params["distance"] = distance
    params["numNodes"] = 10
    params["elecPoolSize"] = 256

    params["fs"] = 100
    params["fsDown"] = 20
    params["fsClassify"] = 2
    params["lag"] = int(params["IEDDURATION"] * params["fs"])
    params["lagDown"] = int(params["IEDDURATION"] * params["fsDown"])

    params["NFOLDS"] = 4
    params["testFold"] = fold

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
    params['BASELINESAVE'] = "base"
    params['BASELINESAVE'] = os.path.join(params["dataFiles"]["ROOT"],
                                      params['BASELINESAVE'])

    for numNodes in range(params["numNodes"]):
        path = os.path.join(params['BASESAVE'], str(numNodes+1))
        os.makedirs(path, exist_ok=True)

    crossValidate(params)