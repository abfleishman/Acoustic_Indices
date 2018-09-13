#!/usr/bin/env python

"""
    Set of functions to compute acoustic indices in the framework of Soundscape Ecology.

    Some features are inspired or ported from those proposed in:
        - seewave R package (http://rug.mnhn.fr/seewave/) / Jerome Sueur, Thierry Aubin and  Caroline Simonis
        - soundecology R package (http://cran.r-project.org/web/packages/soundecology/index.html) / Luis J. Villanueva-Rivera and Bryan C. Pijanowski

    This file use an object oriented type for audio files described in the file "acoustic_index.py".

"""

__author__ = "Patrice Guyot"
__version__ = "0.3"
__credits__ = ["Patrice Guyot", "Alice Eldridge", "Mika Peck"]
__email__ = ["guyot.patrice@gmail.com", "alicee@sussex.ac.uk", "m.r.peck@sussex.ac.uk"]
__status__ = "Development"


from scipy import signal, fftpack
import numpy as np
import matplotlib.pyplot as plt


def TH(file, integer=True):
    """
    Compute Temporal Entropy of Shannon from an audio signal.

    file: an instance of the AudioFile class.
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal. If not, the signal will be set between -1 and 1.

    Ported from the seewave R package.
    """
    if integer:
        sig=file.sig_int
    else:
        sig=file.sig_float

    #env = abs(signal.hilbert(sig)) # Modulo of the Hilbert Envelope
    env = abs(signal.hilbert(sig, fftpack.helper.next_fast_len(len(sig)))) # Modulo of the Hilbert Envelope, computed with the next fast length window

    env = env / np.sum(env)  # Normalization
    N = len(env)
    return - sum([y * np.log2(y) for y in env]) / np.log2(N)


def compute_NDSI(file, windowLength = 1024, anthrophony=[1000,2000], biophony=[2000,11000]):
    """
    Compute Normalized Difference Sound Index from an audio signal.
    This function compute an estimate power spectral density using Welch's method.

    Reference: Kasten, Eric P., Stuart H. Gage, Jordan Fox, and Wooyeong Joo. 2012. The Remote Environ- mental Assessment Laboratory's Acoustic Library: An Archive for Studying Soundscape Ecology. Ecological Informatics 12: 50-67.

    windowLength: the length of the window for the Welch's method.
    anthrophony: list of two values containing the minimum and maximum frequencies (in Hertz) for antrophony.
    biophony: list of two values containing the minimum and maximum frequencies (in Hertz) for biophony.

    Inspired by the seewave R package, the soundecology R package and the original matlab code from the authors.
    """

    #frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=windowLength, noverlap=windowLength/2, nfft=windowLength, detrend=False, return_onesided=True, scaling='density', axis=-1) # Estimate power spectral density using Welch's method
    # TODO change of detrend for apollo
    frequencies, pxx = signal.welch(file.sig_float, fs=file.sr, window='hamming', nperseg=windowLength, noverlap=windowLength/2, nfft=windowLength, detrend='constant', return_onesided=True, scaling='density', axis=-1) # Estimate power spectral density using Welch's method
    avgpow = pxx * frequencies[1] # use a rectangle approximation of the integral of the signal's power spectral density (PSD)
    #avgpow = avgpow / np.linalg.norm(avgpow, ord=2) # Normalization (doesn't change the NDSI values. Slightly differ from the matlab code).

    min_anthro_bin=np.argmin([abs(e - anthrophony[0]) for e in frequencies])  # min freq of anthrophony in samples (or bin) (closest bin)
    max_anthro_bin=np.argmin([abs(e - anthrophony[1]) for e in frequencies])  # max freq of anthrophony in samples (or bin)
    min_bio_bin=np.argmin([abs(e - biophony[0]) for e in frequencies])  # min freq of biophony in samples (or bin)
    max_bio_bin=np.argmin([abs(e - biophony[1]) for e in frequencies])  # max freq of biophony in samples (or bin)

    anthro = np.sum(avgpow[min_anthro_bin:max_anthro_bin])
    bio = np.sum(avgpow[min_bio_bin:max_bio_bin])

    ndsi = (bio - anthro) / (bio + anthro)
    return ndsi


def gini(values):
    """
    Compute the Gini index of values.

    values: a list of values

    Inspired by http://mathworld.wolfram.com/GiniCoefficient.html and http://en.wikipedia.org/wiki/Gini_coefficient
    """
    y = sorted(values)
    n = len(y)
    G = np.sum([i*j for i,j in zip(y,list(range(1,n+1)))])
    G = 2 * G / np.sum(y) - (n+1)
    return G/n


def compute_zcr(file, windowLength=512, windowHop= 256):
    """
    Compute the Zero Crossing Rate of an audio signal.

    file: an instance of the AudioFile class.
    windowLength: size of the sliding window (samples)
    windowHop: size of the lag window (samples)

    return: a list of values (number of zero crossing for each window)
    """


    sig = file.sig_int # Signal on integer values

    times = list(range(0, len(sig)- windowLength +1, windowHop))
    frames = [sig[i:i+windowLength] for i in times]
    return [len(np.where(np.diff(np.signbit(x)))[0])/float(windowLength) for x in frames]


def compute_rms_energy(file, windowLength=512, windowHop=256, integer=False):
    """
    Compute the RMS short time energy.

    file: an instance of the AudioFile class.
    windowLength: size of the sliding window (samples)
    windowHop: size of the lag window (samples)
    integer: if set as True, the Temporal Entropy will be compute on the Integer values of the signal. If not, the signal will be set between -1 and 1.

    return: a list of values (rms energy for each window)
    """
    if integer:
        sig=file.sig_int
    else:
        sig=file.sig_float

    times = list(range(0, len(sig) - windowLength+1, windowHop))
    frames = [sig[i:i + windowLength] for i in times]
    return [np.sqrt(sum([ x**2 for x in frame ]) / windowLength) for frame in frames]


def compute_wave_SNR(file, frame_length_e=512, min_DB=-60, window_smoothing_e=5, activity_threshold_dB=3, hist_number_bins = 100, dB_range = 10, N = 0):
    """

    Computes indices from the Signal to Noise Ratio of a waveform.

    file: an instance of the AudioFile class.
    window_smoothing_e: odd number for sliding mean smoothing of the histogram (can be 3, 5 or 7)
    hist_number_bins - Number of columns in the histogram
    dB_range - dB range to consider in the histogram
    N: The decibel threshold for the waveform is given by the modal intensity plus N times the standard deviation. Higher values of N will remove more energy from the waveform.

    Output:
        Signal-to-noise ratio (SNR): the decibel difference between the maximum envelope amplitude in any minute segment and the background noise.
        Acoustic activity: the fraction of frames within a one minute segment where the signal envelope is more than 3 dB above the level of background noise
        Count of acoustic events: the number of times that the signal envelope crosses the 3 dB threshold
        Average duration of acoustic events: an acoustic event is a portion of recordingwhich startswhen the signal envelope crosses above the 3 dB threshold and ends when it crosses belowthe 3 dB threshold.

    Ref: Towsey, Michael W. (2013) Noise removal from wave-forms and spectro- grams derived from natural recordings of the environment.
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    """



    times = list(range(0, len(file.sig_int)-frame_length_e+1, frame_length_e))
    wave_env = 20*np.log10([np.max(abs(file.sig_float[i : i + frame_length_e])) for i in times])

    minimum = np.max((np.min(wave_env), min_DB)) # If the minimum value is less than -60dB, the minimum is set to -60dB

    hist, bin_edges = np.histogram(wave_env, range=(minimum, minimum + dB_range), bins=hist_number_bins, density=False)


    # hist_smooth = ([np.mean(hist[i - window_smoothing_e/2: i + window_smoothing_e/2]) for i in range(window_smoothing_e/2, len(hist) - window_smoothing_e/2)])
    hist_steps = np.linspace(window_smoothing_e / 2, len(hist) - window_smoothing_e / 2, num=len(hist) - window_smoothing_e + 1)
    hist_smooth = ([np.mean(hist[int(i - window_smoothing_e / 2): int(i + window_smoothing_e / 2)]) for i in hist_steps])
    hist_smooth = np.concatenate((np.zeros(window_smoothing_e//2), hist_smooth, np.zeros(window_smoothing_e//2)))

    modal_intensity = np.argmax(hist_smooth)

    if N>0:
        count_thresh = 68 * sum(hist_smooth) / 100
        count = hist_smooth[modal_intensity]
        index_bin = 1
        while count < count_thresh:
            if modal_intensity + index_bin <= len(hist_smooth):
                count = count + hist_smooth[modal_intensity + index_bin]
            if modal_intensity - index_bin >= 0:
                count = count + hist_smooth[modal_intensity - index_bin]
            index_bin += 1
        thresh = np.min((hist_number_bins, modal_intensity + N * index_bin))
        background_noise = bin_edges[thresh]
    elif N==0:
        background_noise = bin_edges[modal_intensity]

    SNR = np.max(wave_env) - background_noise
    SN = np.array([frame-background_noise-activity_threshold_dB for frame in wave_env])
    acoustic_activity = np.sum([i > 0 for i in SN])/float(len(SN))


    # Compute acoustic events
    start_event = [n[0] for n in np.argwhere((SN[:-1] < 0) & (SN[1:] > 0))]
    end_event = [n[0] for n in np.argwhere((SN[:-1] > 0) & (SN[1:] < 0))]
    if len(start_event)!=0 and len(end_event)!=0:
        if start_event[0]<end_event[0]:
            events=list(zip(start_event, end_event))
        else:
            events=list(zip(end_event, start_event))
        count_acoustic_events = len(events)
        average_duration_e = np.mean([end - begin for begin,end in events] )
        average_duration_s = average_duration_e * file.duration / float(len(SN))
    else:
        count_acoustic_events = 0
        average_duration_s = 0


    dict = {'SNR' : SNR, 'Acoustic_activity' : acoustic_activity, 'Count_acoustic_events' : count_acoustic_events, 'Average_duration' : average_duration_s}
    return dict