"""Generate activation confidence annotations. Taken from MedleyDB code
    https://github.com/marl/medleydb and based on the paper 
   "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research" by
   R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam and J. P. Bello.
"""
import os
import numpy as np
import librosa
import scipy.signal
from scipy.fftpack import hilbert


def compute_activation_confidence(sources, rate=44100, 
                                 win_len=4096, lpf_cutoff=0.075,
                                 theta=0.15, var_lambda=20.0,
                                 amplitude_threshold=0.01, hilb=False):
    """Create the activation confidence annotation for a multitrack. The final
    activation matrix is computed as:
        `C[i, t] = 1 - (1 / (1 + e**(var_lambda * (H[i, t] - theta))))`
    where H[i, t] is the energy of stem `i` at time `t`

    Parameters
    ----------
    mtrack : MultiTrack
        Multitrack object
    win_len : int, default=4096
        Number of samples in each window
    lpf_cutoff : float, default=0.075
        Lowpass frequency cutoff fraction
    theta : float
        Controls the threshold of activation.
    var_labmda : float
        Controls the slope of the threshold function.
    amplitude_threshold : float
        Energies below this value are set to 0.0

    Returns
    -------
    C : np.array
        Array of activation confidence values shape (n_conf, n_stems)
    stem_index_list : list
        List of stem indices in the order they appear in C

    """
    H = []

    # MATLAB equivalent to @hanning(win_len)
    win = scipy.signal.windows.hann(win_len + 2)[1:-1]
    T = sources.shape[1]/rate

    for i in range(sources.shape[0]):
        if hilb:
            H.append(abs(hilbert(sources[i, :])))
        else:
            amp = track_energy(sources[i, :], win_len, win)
            amp = librosa.resample(amp, len(amp)/T, rate)
            H.append(amp)

    # list to numpy array
    H = np.array(H)

    # normalization (to overall energy and # of sources)
    E0 = np.sum(H, axis=0)

    H = H.shape[0] * H / np.max(E0)

    # binary thresholding for low overall energy events
    H[:, E0 < amplitude_threshold] = 0.0

    # LP filter
    b, a = scipy.signal.butter(2, lpf_cutoff, 'low')
    H = scipy.signal.filtfilt(b, a, H, axis=1)

    # logistic function to semi-binarize the output; confidence value
    C = 1.0 - (1.0 / (1.0 + np.exp(np.dot(var_lambda, (H - theta)))))

    # if not hilbert:
    #     T = sources.shape[1]/rate
    #     C = librosa.resample(C, len(C)/T, rate)

    # add time column
    time = librosa.core.frames_to_time(
        np.arange(C.shape[1]), sr=rate, hop_length=win_len // 2
    )

    # stack time column to matrix
    C_out = C

    binary = C
    binary[binary >= 0.5] = 1
    binary[binary <  0.5] = 0

    return C_out, binary, time


def track_energy(wave, win_len, win):
    """Compute the energy of an audio signal

    Parameters
    ----------
    wave : np.array
        The signal from which to compute energy
    win_len: int
        The number of samples to use in energy computation
    win : np.array
        The windowing function to use in energy computation

    Returns
    -------
    energy : np.array
        Array of track energy

    """
    hop_len = win_len // 2

    wave = wave.astype(np.float32)

    wave = np.lib.pad(
        wave, pad_width=(win_len-hop_len, 0), mode='constant', constant_values=0
    )

    # post padding
    wave = librosa.util.fix_length(
        wave, int(win_len * np.ceil(len(wave) / win_len))
    )

    # cut into frames
    wavmat = librosa.util.frame(wave, frame_length=win_len, hop_length=hop_len)

    # Envelope follower
    wavmat = hwr(wavmat) ** 0.5  # half-wave rectification + compression

    return np.mean((wavmat.T * win), axis=1)


def hwr(x):
    """ Half-wave rectification.

    Parameters
    ----------
    x : array-like
        Array to half-wave rectify

    Returns
    -------
    x_hwr : array-like
        Half-wave rectified array

    """
    return (x + np.abs(x)) / 2

