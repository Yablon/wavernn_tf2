import math

import librosa
import numpy as np
import scipy
import tensorflow as tf
from librosa import filters, util
from omegaconf import OmegaConf


def load_wav(path, sample_rate):
  return librosa.core.load(path, sr=sample_rate)[0]


def save_wav(wav, path, sample_rate):
  wav *= 32767 / max(0.01, np.max(np.abs(wav)))
  scipy.io.wavfile.write(path, sample_rate, wav.astype(np.int16))


class AudioProcess:

  def __init__(self, config=None):
    if config:
      self._init_wrapper(config)

  def _init_wrapper(self, config):
    self._config = config
    self.sample_rate = config.sample_rate
    self._preemphasis = config.preemphasis
    self.n_fft = config.n_fft
    self.num_freq = self.n_fft / 2 + 1
    self.hop_length = int(config.frame_shift_ms / 1000 * self.sample_rate)
    self.win_length = int(config.frame_length_ms / 1000 * self.sample_rate)
    self.num_mels = config.num_mels

    self._mel_basis = librosa.filters.mel(self.sample_rate,
                                          self.n_fft,
                                          n_mels=self.num_mels)

    self.fft_window = filters.get_window('hann', self.win_length,
                                         fftbins=True).reshape(
                                             (1, -1)).astype(np.float32)

  def initial(self, config_path):
    self._init_wrapper(OmegaConf.load(config_path).audio_process)

  def load_wav(self, path):
    wave = load_wav(path, self.sample_rate)
    if self._config.rescale:
      return wave / np.abs(wave).max() * 0.97

  def save_wav(self, audio_data, path):
    save_wav(audio_data, path, self.sample_rate)

  def preemphasis(self, audio_data):
    return preemphasis(audio_data, self._preemphasis).astype(np.float32)

  def pad_center(self, audio_data):
    return util.pad_center(audio_data, self.n_fft)

  def frames(self, audio_data):
    audio_data = audio_data.reshape(-1)
    num_points = audio_data.shape[0]
    num_frames = (num_points - self.win_length + 2 * self.hop_length -
                  1) // self.hop_length
    if num_frames <= 0:
      raise RuntimeError('audio data too short')
    frames_points = (num_frames - 1) * self.hop_length + self.win_length
    if frames_points > num_points:
      print('padding frames')
      audio_data = np.pad(audio_data, [0, frames_points - num_points])
    indexer = np.arange(self.win_length)[None, :] + self.hop_length * np.arange(
        num_frames)[:, None]
    return audio_data[indexer]

  def spectrogram(self, audio_data):
    D = self._stft(preemphasis(audio_data, self._preemphasis))
    S = _amp_to_db(np.abs(D)) - self._config.ref_level_db
    return _normalize(S, self._config.min_level_db)

  def normalize(self, stft_results):
    S = _amp_to_db(np.abs(stft_results)) - self._config.ref_level_db
    return _normalize(S, self._config.min_level_db)

  def inv_spectrogram(self, spectrogram):
    '''Converts spectrogram to waveform using librosa'''
    S = _db_to_amp(
        _denormalize(spectrogram, self._config.min_level_db) +
        self._config.ref_level_db)  # Convert back to linear
    return inv_preemphasis(self._griffin_lim(S**self._config.power),
                           self._preemphasis)  # Reconstruct phase

  def melspectrogram(self, y):
    D = self._stft(preemphasis(y, self._preemphasis))
    S = _amp_to_db(self._linear_to_mel(np.abs(D))) - self._config.ref_level_db
    return _normalize(S, self._config.min_level_db)

  def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(self.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
      if np.max(wav[x:x + window_length]) < threshold:
        return x + hop_length
    return len(wav)

  def _griffin_lim(self, S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = self._istft(S_complex * angles)
    for i in range(self._config.griffin_lim_iters):
      angles = np.exp(1j * np.angle(self._stft(y)))
      y = self._istft(S_complex * angles)
    return y

  def _linear_to_mel(self, spectrogram):
    return np.dot(self._mel_basis, spectrogram)

  def _stft(self, y):
    return librosa.stft(y=y,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        pad_mode='constant')

  def _istft(self, y):
    return librosa.istft(y,
                         hop_length=self.hop_length,
                         win_length=self.win_length)


def _amp_to_db(x):
  return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
  return np.power(10.0, x * 0.05)


def _db_to_amp_tensorflow(x):
  return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)


def _normalize(S, min_level_db):
  return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def _denormalize(S, min_level_db):
  return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def preemphasis(audio_data, preemphasis_coef):
  return scipy.signal.lfilter([1, -preemphasis_coef], [1], audio_data)


def inv_preemphasis(audio_data, preemphasis_coef):
  return scipy.signal.lfilter([1], [1, -preemphasis_coef], audio_data)
