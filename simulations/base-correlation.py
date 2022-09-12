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
import pickle
import os
import sys

import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format='retina'
import seaborn as sns
import tikzplotlib

import numpy as np
import pandas as pd

sys.path.append('../library')
import IedDetector 

# +
ROOT = '/users/sista/jdan/miniEEG/base'

subjects = ['patient003', 'patient004', 'patient009', 'patient013',
            'patient020', 'patient021', 'patient025', 'patient030',
            'patient036', 'patient037', 'patient040', 'patient047',
            'patient050', 'patient051', 'patient080']

results = dict()
results['subject'] = []

keys = ['auc', 'auc10',
        'snr', 'snrMaxSnr', 'signal', 'noise', 'signalMaxSnr', 'noiseMaxSnr']
for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
    keys.append("threshold-{}".format(t))
    keys.append("fp-{}".format(t))
    keys.append("f1-{}".format(t))
    keys.append("cohenKappa-{}".format(t))
    keys.append("cohenKappa10-{}".format(t))
    keys.append("cor-{}".format(t))

for key in keys:
    results[key] = []
    
results['r'] = []

toRun = list()
    
for subject in subjects:
    filename = os.path.join(ROOT, 'simulation-{}.pkl'.format(subject))
    if os.path.exists(filename):
        file = open(filename,'rb')
        result = pickle.load(file)
        file.close()    

        results['subject'].append(subject)

        for key in keys:
            results[key].append(result['results'][key])

        results['r'].append(result['results']['snrMaxSnr']/result['results']['snr'])
    else:
        # print('{} {} {} / {}'.format(subject, distance, fold, numNodes))
        toRun.append('{} {} {}'.format(subject, distance, fold))
results = pd.DataFrame(results)
# -

plt.figure(figsize=(16, 6))
sns.histplot(x='cor-0.6', data=results,
            binwidth=0.10, binrange=[0,1])
tikzplotlib.save("correlation.tex")
plt.show()

np.median(results['cor-0.6'])

np.sum(results['fp-0.6'])+0.6*4088

plt.figure(figsize=(16, 6))
sns.histplot(x='cohenKappa10-0.6', data=results,
            binwidth=0.10, binrange=[0,1])
plt.show()

np.median(results['cohenKappa10-0.6'])

results[results['cor-0.6'] < 0.9][['subject', 'cor-0.6']]


