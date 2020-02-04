import maplotlib.pyplot as plt
import seaborn as sns

def plot_lofargram(Sxx, freq, time, ax=None, figsize=(10, 10), savepath=None):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    ax.imshow()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
    

