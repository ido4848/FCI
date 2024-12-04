import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(3, 2))
ax = fig.subplots(1, 1)

# auc ranges from 0.9 to 0.999
auc = np.linspace(0.9, 0.999, 100)
mult = 1000
minimal = 0.9
maximal = 0.999
fci = np.log10(mult*(1-auc)) / np.log10(1000*(1-minimal))

ax.plot(auc, fci, color='black')
ax.set_xlabel('AUC')
ax.set_ylabel('FCI')

ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])

ax.scatter(0.9, 1.0, color='red', s=20)
ax.scatter(0.999, 0.0, color='red', s=20)

ax.set_title('AUC vs FCI')