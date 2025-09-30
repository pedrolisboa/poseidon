import os
import numpy as np

class SpectrogramLoader:
    """
    A stateless loader for a single processed spectrogram, designed for use
    within a PyTorch DataLoader.

    It holds a filepath and loads the data directly from disk on every request,
    preventing any long-term caching within Python's memory.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self._shape = None # We can cache the shape, as it's small.

    def __getitem__(self, key):
        """Loads data from disk and returns the requested array."""
        if key != 'sxx':
            raise KeyError(f"SpectrogramLoader only supports fetching 'sxx', not '{key}'")
        # 'with' statement ensures the file handle is properly closed.
        with np.load(self.filepath) as data:
            return data[key]

    @property
    def shape(self):
        """Gets the shape of the 'sxx' array without holding it in memory."""
        if self._shape is None:
            # Load the data just once to get the shape, then discard.
            sxx = self['sxx']
            self._shape = sxx.shape
        return self._shape

    def __repr__(self):
        return f"<SpectrogramLoader(file='{os.path.basename(self.filepath)}')>"