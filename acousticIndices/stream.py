import struct
from datetime import datetime
import numpy as np
from multiprocessing import Process

from cached_property import cached_property
import pyaudio
from scipy.io.wavfile import read as wavread

from acousticIndices.acoustic_index import pcm2float, float2pcm

def stream_to_np(frame):
    block = frame
    countB = len(block) / 2
    format = "%dh" % (countB)
    shorts = struct.unpack(format, block)
    curWindow = np.array(shorts, dtype=np.int16)
    return curWindow

class AudioChunk:
    def __init__(self, samplerate, data=None):
        self.samplerate = samplerate
        self.duration = 0 #TODO

        self.data = data

    def filter(self, filter_fn):
        """Filter the audio data

        Parameters
        ----------
        filter_fn : function_like
            Accepts a single argument, the result of this chunk's .as_float property

        Returns
        -------
        AudioChunk
            A filtered version of this AudioChunk

        """
        filtered_float = filter_fn(self.as_float)
        filtered_int = float2pcm(filtered_float)

        return AudioChunk.from_np_array(filtered_int, self.samplerate)

    @property
    def niquist(self):
        return self.nyquist

    @property
    def nyquist(self):
        return self.samplerate/2

    @property
    def as_int(self):
        return self.data

    @cached_property
    def as_float(self):
        return pcm2float(self.data, dtype='float64')

    @classmethod
    def from_np_array(cls, np_array, samplerate=44100):
        return cls(samplerate, np_array)

    @classmethod
    def from_pyaudio_stream(cls, stream, seconds):
        frames_per_buffer = 8820

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=44100,
                         input=True,
                         frames_per_buffer=frames_per_buffer)

        frames = []

        start_time = datetime.now()
        while (datetime.now()-start_time).total_seconds() < seconds:
            if stream.get_read_available() >= frames_per_buffer:
                frame = stream.read(frames_per_buffer) # self.frames_per_buffer
                frames.append(frame)

        chunk = b''.join(frames)
        chunk_np = stream_to_np(chunk)

        return cls(44100, chunk_np)

    @classmethod
    def from_wav_file(cls, file_path):
        try:
            sr, sig = wavread(file_path)
        except IOError:
            print("Error: can\'t read the audio file:", file_path)

        return cls(sr, sig)

class AudioCapture(Process):

    def __init__(self, queue, stop_event, blocksize, samplingrate,
                 format=pyaudio.paInt16, rate=16000,  #frames_per_buffer=3200,
                 record=False
                 ):
        super(AudioCapture, self).__init__()

        self.queue = queue
        self.stop_event = stop_event

        self.blocksize = blocksize
        self.samplingrate = samplingrate

        self.format = format
        self.rate = rate
        self.frames_per_buffer = int(self.rate * self.blocksize)

        self.record = record

    def run(self):
        pa = pyaudio.PyAudio()

        stream = pa.open(format=self.format,
                         channels=1,
                         rate=self.rate,
                         input=True,
                         frames_per_buffer=self.frames_per_buffer)

        frames = []

        while True:
            frame = stream.read(self.frames_per_buffer) # self.frames_per_buffer
            frames.append(frame)

            if self.stop_event.is_set():
                print("Stopping evaluation.")
                self.stop_event.clear()
                self.stop()

                chunk = b''.join(frames)
                chunk_np = stream_to_np(chunk)

                achunk = AudioChunk.from_np_array(chunk_np)

                self.queue.put(achunk)

                # if self.record:
                #     waveFile = wave.open(datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.wav', 'wb')
                #     waveFile.setnchannels(1)
                #     waveFile.setsampwidth(pa.get_sample_size(self.format))
                #     waveFile.setframerate(self.rate)
                #     waveFile.writeframes(chunk)
                #     waveFile.close()

                break


    def stop(self):
        self.go = False