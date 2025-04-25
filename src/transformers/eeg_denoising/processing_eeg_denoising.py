import numpy as np
import mne

class EEGDenoisingProcessor:
    def __init__(self, notch_freq=60., bandpass=(0.5, 50.)):
        self.notch_freq = notch_freq
        self.bandpass = bandpass

    def process(self, raw_eeg):
        raw = mne.io.RawArray(raw_eeg, mne.create_info(raw_eeg.shape[0], sfreq=256))
        raw.notch_filter(self.notch_freq)
        raw.filter(*self.bandpass)
        return raw.get_data()