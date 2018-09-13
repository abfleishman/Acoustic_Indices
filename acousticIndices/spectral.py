import numpy as np
import pyaudio
from scipy import signal

from acousticIndices.index import gini
from acousticIndices.stream import AudioChunk


def spectrogram(file, windowLength: int=512, windowHop= 256, scale_audio=True, square=True, windowType='hanning', centered=False, normalized = False ):
    """
    Compute a spectrogram of an audio signal.
    Return a list of list of values as the spectrogram, and a list of frequencies.

    Keyword arguments:
    file -- the real part (default 0.0)

    Parameters:
    file: an instance of the AudioFile class.
    windowLength: length of the fft window (in samples)
    windowHop: hop size of the fft window (in samples)
    scale_audio: if set as True, the signal samples are scale between -1 and 1 (as the audio convention). If false the signal samples remains Integers (as output from scipy.io.wavfile)
    square: if set as True, the spectrogram is computed as the square of the magnitude of the fft. If not, it is the magnitude of the fft.
    hamming: if set as True, the spectrogram use a correlation with a hamming window.
    centered: if set as true, each resulting fft is centered on the corresponding sliding window
    normalized: if set as true, divide all values by the maximum value
    """

    if scale_audio:
        sig = file.as_float # use signal with float between -1 and 1
    else:
        sig = file.as_int # use signal with integers


    W = signal.get_window(windowType, windowLength, fftbins=False)

    if centered:
        time_shift = int(windowLength/2)
        times = list(range(time_shift, len(sig)+1-time_shift, windowHop)) # centered
        frames = [sig[i-time_shift:i+time_shift]*W for i in times] # centered frames
    else:
        times = list(range(0, len(sig)-windowLength+1, windowHop))
        frames = [sig[i:i+windowLength]*W for i in times]

    if square:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:windowLength//2]**2 for frame in frames]
    else:
        spectro =  [abs(np.fft.rfft(frame, windowLength))[0:windowLength//2] for frame in frames]

    spectro=np.transpose(spectro) # set the spectro in a friendly way

    if normalized:
        with np.errstate(divide='raise', invalid='raise'):
            try:
                spectro = spectro/np.max(spectro) # set the maximum value to 1 y
            except Exception as e:
                pass


    frequencies = [e * file.niquist / float(windowLength / 2) for e in range(windowLength // 2)] # vector of frequency<-bin in the spectrogram
    return spectro, frequencies


def ACI(spectro,j_bin):
    """
    Compute the Acoustic Complexity Index from the spectrogram of an audio signal.

    Reference: Pieretti N, Farina A, Morri FD (2011) A new methodology to infer the singing activity of an avian community: the Acoustic Complexity Index (ACI). Ecological Indicators, 11, 868-873.

    Ported from the soundecology R package.

    spectro: the spectrogram of the audio signal
    j_bin: temporal size of the frame (in samples)


    """

    #times = range(0, spectro.shape[1], j_bin) # relevant time indices
    times = list(range(0, spectro.shape[1]-10, j_bin)) # alternative time indices to follow the R code

    jspecs = [np.array(spectro[:,i:i+j_bin]) for i in times]  # sub-spectros of temporal size j

    aci = [sum((np.sum(abs(np.diff(jspec)), axis=1) / np.sum(jspec, axis=1))) for jspec in jspecs] 	# list of ACI values on each jspecs
    main_value = sum(aci)
    temporal_values = aci

    return main_value, temporal_values # return main (global) value, temporal values


def BI(spectro, frequencies, min_freq = 2000, max_freq = 8000):
    """
    Compute the Bioacoustic Index from the spectrogram of an audio signal.
    In this code, the Bioacoustic Index correspond to the area under the mean spectre (in dB) minus the minimum frequency value of this mean spectre.

    Reference: Boelman NT, Asner GP, Hart PJ, Martin RE. 2007. Multi-trophic invasion resistance in Hawaii: bioacoustics, field surveys, and airborne remote sensing. Ecological Applications 17: 2137-2144.

    spectro: the spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    min_freq: minimum frequency (in Hertz)
    max_freq: maximum frequency (in Hertz)

    Ported from the soundecology R package.
    """

    min_freq_bin = int(np.argmin([abs(e - min_freq) for e in frequencies])) # min freq in samples (or bin)
    max_freq_bin = int(np.ceil(np.argmin([abs(e - max_freq) for e in frequencies]))) # max freq in samples (or bin)

    min_freq_bin = min_freq_bin - 1 # alternative value to follow the R code



    spectro_BI = 20 * np.log10(spectro/np.max(spectro))  #  Use of decibel values. Equivalent in the R code to: spec_left <- spectro(left, f = samplingrate, wl = fft_w, plot = FALSE, dB = "max0")$amp
    spectre_BI_mean = 10 * np.log10 (np.mean(10 ** (spectro_BI/10), axis=1))     # Compute the mean for each frequency (the output is a spectre). This is not exactly the mean, but it is equivalent to the R code to: return(a*log10(mean(10^(x/a))))
    spectre_BI_mean_segment =  spectre_BI_mean[min_freq_bin:max_freq_bin]   # Segment between min_freq and max_freq
    spectre_BI_mean_segment_normalized = spectre_BI_mean_segment - min(spectre_BI_mean_segment) # Normalization: set the minimum value of the frequencies to zero.
    area = np.sum(spectre_BI_mean_segment_normalized / (frequencies[1]-frequencies[0]))   # Compute the area under the spectre curve. Equivalent in the R code to: left_area <- sum(specA_left_segment_normalized * rows_width)

    return area


def SH(spectro):
    """
    Compute Spectral Entropy of Shannon from the spectrogram of an audio signal.

    spectro: the spectrogram of the audio signal

    Ported from the seewave R package.
    """
    N = spectro.shape[0]
    spec = np.sum(spectro,axis=1)
    spec = spec / np.sum(spec)  # Normalization by the sum of the values
    main_value = - sum([y * np.log2(y) for y in spec]) / np.log2(N)  #Equivalent in the R code to: z <- -sum(spec*log(spec))/log(N)
    #temporal_values = [- sum([y * np.log2(y) for y in frame]) / (np.sum(frame) * np.log2(N)) for frame in spectro.T]
    return main_value


def AEI(spectro, freq_band_Hz, max_freq=10000, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Evenness Index of an audio signal.

    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute AEI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute AEI (in Hertz)

    Ported from the soundecology R package.
    """

    bands_Hz = list(range(0, max_freq, freq_step))
    bands_bin = [f // freq_band_Hz for f in bands_Hz]

    spec_AEI = 20*np.log10(spectro/np.max(spectro))
    spec_AEI_bands = [spec_AEI[bands_bin[k]:bands_bin[k]+bands_bin[1],] for k in range(len(bands_bin))]

    values = [np.sum(spec_AEI_bands[k]>db_threshold)/float(spec_AEI_bands[k].size) for k in range(len(bands_bin))]

    return gini(values)


def ADI(spectro, freq_band_Hz,  max_freq=10000, db_threshold=-50, freq_step=1000):
    """
    Compute Acoustic Diversity Index.

    Reference: Villanueva-Rivera, L. J., B. C. Pijanowski, J. Doucette, and B. Pekin. 2011. A primer of acoustic analysis for landscape ecologists. Landscape Ecology 26: 1233-1246.

    spectro: spectrogram of the audio signal
    freq_band_Hz: frequency band size of one bin of the spectrogram (in Hertz)
    max_freq: the maximum frequency to consider to compute ADI (in Hertz)
    db_threshold: the minimum dB value to consider for the bins of the spectrogram
    freq_step: size of frequency bands to compute ADI (in Hertz)


    Ported from the soundecology R package.
    """


    bands_Hz = list(range(0, max_freq, freq_step))
    bands_bin = [f // freq_band_Hz for f in bands_Hz]

    spec_ADI = 20*np.log10(spectro/np.max(spectro))
    spec_ADI_bands = [spec_ADI[bands_bin[k]:bands_bin[k]+bands_bin[1],] for k in range(len(bands_bin))]

    #TODO is this valid?
    bin_k = filter(lambda k: spec_ADI_bands[k].size>0, range(len(bands_bin)))
    values = [np.sum(spec_ADI_bands[k]>db_threshold)/float(spec_ADI_bands[k].size) for k in bin_k]

    # Shannon Entropy of the values
    #shannon = - sum([y * np.log(y) for y in values]) / len(values)  # Follows the R code. But log is generally log2 for Shannon entropy. Equivalent to shannon = False in soundecology.

    # The following is equivalent to shannon = True (default) in soundecology. Compute the Shannon diversity index from the R function diversity {vegan}.
    #v = [x/np.sum(values) for x in values]
    #v2 = [-i * j  for i,j in zip(v, np.log(v))]
    #return np.sum(v2)

    # Remove zero values (Jan 2016)
    values = [value for value in values if value != 0]

    #replace zero values by 1e-07 (closer to R code, but results quite similars)
    #values = [x if x != 0 else 1e-07 for x in values]

    return np.sum([-i/ np.sum(values) * np.log(i / np.sum(values))  for i in values])


def spectral_centroid(spectro, frequencies):
    """
    Compute the spectral centroid of an audio signal from its spectrogram.

    spectro: spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    """

    centroid = [ np.sum(magnitudes*frequencies) / np.sum(magnitudes) for magnitudes in spectro.T]


    return centroid

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def remove_noiseInSpectro(spectro, histo_relative_size=8, window_smoothing=5, N=0.1, dB=False, plot=False):
    """

    Compute a new spectrogram which is "Noise Removed".

    spectro: spectrogram of the audio signal
    histo_relative_size: ration between the size of the spectrogram and the size of the histogram
    window_smoothing: number of points to apply a mean filtering on the histogram and on the background noise curve
    N: Parameter to set the threshold around the modal intensity
    dB: If set at True, the spectrogram is converted in decibels
    plot: if set at True, the function plot the orginal and noise removed spectrograms

    Output:
        Noise removed spectrogram

    Ref: Towsey, Michael W. (2013) Noise removal from wave-forms and spectrograms derived from natural recordings of the environment.
    Towsey, Michael (2013), Noise Removal from Waveforms and Spectrograms Derived from Natural Recordings of the Environment. Queensland University of Technology, Brisbane.
    """

    low_value = 1.e-07 # Minimum value for the new spectrogram (preferably slightly higher than 0)

    if dB:
        spectro = 20*np.log10(spectro)

    len_spectro_e = len(spectro[0])
    histo_size = len_spectro_e//histo_relative_size

    background_noise=[]
    for row in spectro:
        hist, bin_edges = np.histogram(row, bins=histo_size, density=False)

        # hist_smooth = ([np.mean(hist[i - window_smoothing /2: i + window_smoothing /2]) for i in range(window_smoothing /2, len(hist) - window_smoothing /2)])
        hist_steps = np.linspace(window_smoothing / 2, len(hist) - window_smoothing / 2, num=len(hist) - window_smoothing + 1)
        hist_smooth = ([np.mean(hist[int(i - window_smoothing / 2): int(i + window_smoothing / 2)]) for i in hist_steps])
        hist_smooth = np.concatenate((np.zeros(window_smoothing//2), hist_smooth, np.zeros(window_smoothing //2)))


        modal_intensity = int(np.min([np.argmax(hist_smooth), 95 * histo_size / 100])) # test if modal intensity value is in the top 5%

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
            thresh = int(np.min((histo_size, modal_intensity + N * index_bin)))
            background_noise.append(bin_edges[thresh])
        elif N==0:
            background_noise.append(bin_edges[modal_intensity])


    # background_noise_smooth = ([np.mean(background_noise[i - window_smoothing /2: i + window_smoothing /2]) for i in range(window_smoothing /2, len(background_noise) - window_smoothing /2)])
    bn_steps = np.linspace(window_smoothing / 2, len(background_noise) - window_smoothing / 2, num=len(background_noise) - window_smoothing + 1)
    background_noise_smooth = ([np.mean(background_noise[int(i - window_smoothing / 2): int(i + window_smoothing / 2)]) for i in bn_steps])
    # keep background noise at the end to avoid last row problem (last bin with old microphones)
    background_noise_smooth = np.concatenate((background_noise[0:(window_smoothing//2)], background_noise_smooth, background_noise[-(window_smoothing//2):]))

    new_spec = np.array([col - background_noise_smooth for col in spectro.T]).T
    new_spec = new_spec.clip(min=low_value) # replace negative values by value close to zero

    #Figure
    if plot:
        colormap="jet"
        fig = plt.figure()
        a = fig.add_subplot(1,2,1)
        if dB:
            plt.imshow(new_spec, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        else:
            plt.imshow(20*np.log10(new_spec), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        a = fig.add_subplot(1,2,2)
        if dB:
            plt.imshow(new_spec, origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        else:
            plt.imshow(20*np.log10(spectro), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        plt.show()



    return new_spec


def NB_peaks(spectro, frequencies, freqband = 200, normalization= True, slopes=(0.01,0.01)):
    """

    Counts the number of major frequency peaks obtained on a mean spectrum.

    spectro: spectrogram of the audio signal
    frequencies: list of the frequencies of the spectrogram
    freqband: frequency threshold parameter (in Hz). If the frequency difference of two successive peaks is less than this threshold, then the peak of highest amplitude will be kept only.
    normalization: if set at True, the mean spectrum is scaled between 0 and 1
    slopes: amplitude slope parameter, a tuple of length 2. Refers to the amplitude slopes of the peak. The first value is the left slope and the second value is the right slope. Only peaks with higher slopes than threshold values will be kept.

    Ref: Gasc, A., Sueur, J., Pavoine, S., Pellens, R., & Grandcolas, P. (2013). Biodiversity sampling using a global acoustic approach: contrasting sites with microendemics in New Caledonia. PloS one, 8(5), e65311.

    """

    meanspec = np.array([np.mean(row) for row in spectro])

    if normalization:
         meanspec =  meanspec/np.max(meanspec)

    # Find peaks (with slopes)
    peaks_indices = np.r_[False, meanspec[1:] > np.array([x + slopes[0] for x in meanspec[:-1]])] & np.r_[meanspec[:-1] > np.array([y + slopes[1] for y in meanspec[1:]]), False]
    peaks_indices = peaks_indices.nonzero()[0].tolist()

    #peaks_indices = signal.argrelextrema(np.array(meanspec), np.greater)[0].tolist() # scipy method (without slope)


    # Remove peaks with difference of frequency < freqband
    nb_bin=next(i for i,v in enumerate(frequencies) if v > freqband) # number of consecutive index
    for consecutiveIndices in [np.arange(i, i+nb_bin) for i in peaks_indices]:
        if len(np.intersect1d(consecutiveIndices,peaks_indices))>1:
            # close values has been found
            maxi = np.intersect1d(consecutiveIndices,peaks_indices)[np.argmax([meanspec[f] for f in np.intersect1d(consecutiveIndices,peaks_indices)])]
            peaks_indices = [x for x in peaks_indices if x not in consecutiveIndices] # remove all inddices that are in consecutiveIndices
            peaks_indices.append(maxi) # append the max
    peaks_indices.sort()


    peak_freqs = [frequencies[p] for p in peaks_indices] # Frequencies of the peaks

    return len(peaks_indices)

def main(args):

    pa = pyaudio.PyAudio()
    FORMAT = pyaudio.paInt16

    midTermBufferSize = int(args.samplingrate * args.blocksize)

    print(midTermBufferSize)


    stream = pa.open(format=FORMAT,
                 channels=1,
                 rate=args.samplingrate,
                 input=True,
                 frames_per_buffer=midTermBufferSize)
    audio_chunk = AudioChunk.from_pyaudio_stream(stream, args.time)

    # Filtering
    freq_filter = 300
    Wn = freq_filter / float(audio_chunk.niquist)
    order = 8
    [b, a] = signal.butter(order, Wn, btype='highpass')
    filter_fn = lambda x: signal.filtfilt(b, a, x)
    filtered_audio = audio_chunk.filter(filter_fn)

    # Acoustic Complexity
    spectro_norm, _ = spectrogram(filtered_audio, windowLength=512, windowHop=512, scale_audio=False, square=False,
                        windowType='hamming', centered=False, normalized=True)
    j_bin = 5 * filtered_audio.samplerate // 512  # transform j_bin in samples
    main_value, temporal_values = ACI(spectro_norm, j_bin)
    print("ACI: ", np.average(temporal_values))

    # Acoustic Diversity
    freq_band_Hz = 10000 // 1000
    windowLength = filtered_audio.samplerate // freq_band_Hz
    spectro, _ = spectrogram(filtered_audio, windowLength=windowLength, windowHop=windowLength, scale_audio=True,
                                     square=False, windowType='hanning', centered=False, normalized=False)
    main_value = ADI(spectro, freq_band_Hz, max_freq=10000, db_threshold=-50, freq_step=1000)
    print("ADI: ", main_value)

    # Acoustic Evenness
    main_value = AEI(spectro, freq_band_Hz, max_freq=10000, db_threshold=-50, freq_step=1000)
    print("AEI: ", main_value)

    # Spectral Entropy (Shannon)
    spectro, _ = spectrogram(filtered_audio, windowLength=512, windowHop=256, scale_audio=True, square=False,
                        windowType='hamming', centered=False, normalized=False)
    main_value = SH(spectro)
    print("Entropy: ", main_value)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Real time audio analysis")
    parser.add_argument("-bs", "--blocksize", type=float, choices=[0.1, 0.2, 0.3, 0.4, 0.5], default=0.30,
                        help="Recording block size")
    parser.add_argument("-fs", "--samplingrate", type=int, choices=[4000, 8000, 16000, 32000, 44100], default=44100,
                        help="Recording block size")
    parser.add_argument("--chromagram", action="store_true", help="Show chromagram")
    parser.add_argument("--spectrogram", action="store_true", help="Show spectrogram")
    parser.add_argument("--recordactivity", action="store_true", help="Record detected sounds to wavs")
    parser.add_argument('-t', "--time", type=float, help="Time (seconds) to record", default=3.0)
    args = parser.parse_args()

    main(args)