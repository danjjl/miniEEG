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
subjects = ['patient003', 'patient004', 'patient009', 'patient013',
            'patient020', 'patient021', 'patient025', 'patient030',
            'patient036', 'patient047', 'patient049', 'patient050',
            'patient051', 'patient080']
distances = [2, 3.5, 5, 6.5, 8]
nfolds = 4

f = open("args.txt", "w")
for subject in subjects:
    for distance in distances:
        for fold in range(nfolds):
            f.write("{} {} {}\n".format(subject, distance, fold))
f.close()
# -


