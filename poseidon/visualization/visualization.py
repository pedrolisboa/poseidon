import matplotlib.pyplot as plt
import seaborn as sns

def plot_lofargram(Sxx, freq, time, ax=None, figsize=(10, 10), savepath=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()


    # Use extent to set the coordinates of the image boundaries
    # The origin='lower' argument ensures that the y-axis (time) increases from bottom to top.
    ax.imshow(Sxx, aspect='auto', origin='lower', 
              extent=[freq[0], freq[-1], time[0], time[-1]])

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_title('LOFARgram')

    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight')
        
    return ax
    