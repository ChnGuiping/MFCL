
import numpy as np
import random

import torch
from scipy.signal import resample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        #print(seq.shape)
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class AddGaussianSNR(object):
    def __init__(self, min_snr=3, max_snr=30):
        self.min_snr = min_snr
        self.max_snr = max_snr

    def __call__(self, signal: np.ndarray):
        if np.random.rand() > 0.2:
            snr = np.random.randint(self.min_snr, self.max_snr)
            clear_rms = np.sqrt(np.mean(np.square(signal)))  # Use numpy operations
            noise_rms = clear_rms / (10**(snr / 20))
            noise = np.random.normal(0.0, noise_rms, size=signal.shape)
            return signal + noise
        else:
            return signal


class Shift(object):
    def __init__(self, max_shift_factor=0.3):
        self.max_shift_factor = max_shift_factor

    def __call__(self, signal: np.ndarray):
        if np.random.rand() > 0.7:
            shift_length = int(signal.shape[-1] * self.max_shift_factor)
            num_places_to_shift = np.random.randint(-shift_length, shift_length)
            return np.roll(signal, num_places_to_shift, axis=-1)  # Use numpy's roll
        else:
            return signal


class TimeMask(object):
    def __init__(self, min_mask_factor=0.1, max_mask_factor=0.45):
        self.min_mask_factor = min_mask_factor
        self.max_mask_factor = max_mask_factor

    def __call__(self, signal: np.ndarray):
        if np.random.rand() > 0.5:
            signal_length = signal.shape[-1]
            signal_copy = np.copy(signal)  # Use numpy copy
            mask_factor = np.random.uniform(self.min_mask_factor, self.max_mask_factor)
            mask_length = int(signal_length * mask_factor)
            mask_start = np.random.randint(0, signal_length - mask_length + 1)
            signal_copy[..., mask_start:mask_start + mask_length] = 0
            return signal_copy
        else:
            return signal


class Fade(object):
    def __init__(self, fade_in_factor=0.3, fade_out_factor=0.3):
        self.fade_in_factor = fade_in_factor
        self.fade_out_factor = fade_out_factor

    def __call__(self, signal: np.ndarray):
        if np.random.rand() > 0.8:
            signal_length = signal.shape[-1]
            fade_in_length = int(self.fade_in_factor * signal_length)
            fade_out_length = int(self.fade_out_factor * signal_length)
            signal_copy = np.copy(signal)  # Use numpy copy

            # Apply fade-in effect
            if np.random.rand() > 0.5:
                fade_in = np.linspace(0, 1, fade_in_length)
                fade_in = np.log10(0.1 + fade_in) + 1
                fade_in_mask = np.concatenate((fade_in, np.ones(signal_length - fade_in_length)))
                signal_copy *= fade_in_mask

            # Apply fade-out effect
            if np.random.rand() > 0.5:
                fade_out = np.linspace(1, 0, fade_out_length)
                fade_out = np.log10(0.1 + fade_out) + 1
                fade_out_mask = np.concatenate((np.ones(signal_length - fade_out_length), fade_out))
                signal_copy *= fade_out_mask

            return signal_copy
        else:
            return signal


class Gain(object):
    def __init__(self, gain_min=0.5, gain_max=1.5):
        self.gain_min = gain_min
        self.gain_max = gain_max

    def __call__(self, signal: np.ndarray):
        if np.random.rand() > 0.2:
            gain_factor = np.random.uniform(self.gain_min, self.gain_max)
            return signal * gain_factor  # Use numpy's element-wise multiplication
        else:
            return signal


class Flip(object):
    def __call__(self, signal: np.ndarray):
        if np.random.rand() > 0.2:
            if len(signal.shape) == 1:
                return np.flip(signal)  # Use numpy's flip for 1D arrays
            elif len(signal.shape) == 2:
                return np.flip(signal, axis=1)  # Use numpy's flip for 2D arrays
            else:
                raise Exception(f"Signal with shape {signal.shape} is not supported for flipping.")
        else:
            return signal



class Normalize(object):
    def __init__(self, type = "0-1"): # "0-1","-1-1","mean-std"
        self.type = type

    def __call__(self, seq):
        if  self.type == "0-1":
            seq =(seq-seq.min())/(seq.max()-seq.min())
        elif  self.type == "-1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std" :
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')

        return seq


class TwoStrongTransform(object):
    def __init__(self):
        self.original = Compose([
            Reshape(), Normalize("0-1"), Retype()
        ])

        self.strong1 = Compose([
            Reshape(), AddGaussianSNR(), TimeMask(), Gain(), Shift(), Flip(), Fade(), Normalize("0-1"), Retype()
        ])

        self.strong2 = Compose([
            Reshape(), AddGaussianSNR(), TimeMask(), Gain(), Shift(), Flip(), Fade(), Normalize("0-1"), Retype()
        ])

    def __call__(self, x):
        original = self.original(x)
        strong1 = self.strong1(x)
        strong2 = self.strong2(x)
        return original, strong1, strong2