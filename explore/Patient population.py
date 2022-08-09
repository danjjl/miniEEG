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
import glob
import os

import pandas as pd
from IPython.display import display, Markdown

# +
ROOT = '/users/sista/jdan/miniEEG/Persyst'

subjects = glob.glob(os.path.join(ROOT, 'patient*.xlsx'))
for i, subject in enumerate(subjects):
    subjects[i] = subject.split('/')[-1][:-5]
    
def load_spikes(confidence):
    spikes = {
    'subject' : [],
    'confidence' : [],
    'amplitude' : [],
    'channel' : []
    }
    for subject in subjects:
        df = pd.read_excel(os.path.join(ROOT, subject + '.xlsx'))
        x = df[df.Amplitude < 150]
        x = x[x.Perception > confidence]
        df = x.groupby('Channel').filter(lambda grp: grp.Channel.count() > 5)

        spikes['confidence'] += list(df.Perception)
        spikes['amplitude'] += list(df.Amplitude)
        spikes['channel'] += list(df.Channel)
        spikes['subject'] += [subject]*len(df)

    spikes = pd.DataFrame(spikes)
    
    return spikes
    
    

display(Markdown('### High'))
spikes = load_spikes(0.9)
display(spikes.subject.value_counts())


display(Markdown('### Medium'))
spikes = load_spikes(0.4)
display(spikes.subject.value_counts())

display(Markdown('### Low'))
spikes = load_spikes(0.1)
display(spikes.subject.value_counts())

spikes = load_spikes(0.9)
# -

x = spikes[spikes.confidence > 0.9].groupby(['subject']).count()
subjects = list(x[x.confidence > 10].confidence.keys())
print(len(subjects))
print(subjects)

len(glob.glob(os.path.join(ROOT, 'patient*.xlsx')))


