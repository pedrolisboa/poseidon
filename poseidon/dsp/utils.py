import scipy.signal
from fractions import Fraction

def resample(signal, fs, final_fs, window=('kaiser', 5.0)):
    resample_ratio = Fraction(final_fs, fs)

    upsampling_factor = resample_ratio.numerator
    downsampling_factor = resample_ratio.denominator

    if upsampling_factor == 1:
        return scipy.signal.decimate(
            scipy, 
            downsampling_factor, 
            'fir', 
            zero_phase=True
        )

    resampled_signal = scipy.signal.resample_poly(
        signal, 
        upsampling_factor, 
        downsampling_factor,
        axis=0, 
        window=window
    )

    return resampled_signal


def freq_bins_cutoff(Sxx, fs, target_fs):
    raise NotImplementedError