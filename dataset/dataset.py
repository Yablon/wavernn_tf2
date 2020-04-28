from pathlib import Path

import tensorflow as tf

from dataset.audio import AudioProcess
import scipy.fftpack as fft

import numpy as np


def read_metadata(meta_path: Path) -> list:
  meta_lines = meta_path.open('r', encoding='utf-8').readlines()
  wav_names = [line.split("|")[0] for line in meta_lines]
  return [
      meta_path.parent.joinpath('wavs', wav_name + '.wav').as_posix()
      for wav_name in wav_names
  ]


class WaveRNNLoader:

  def __init__(self, config):
    self.audio_processor = AudioProcess(config)

  def random_desired_samples(self, path):
    audio_data = self.audio_processor.load_wav(path)
    audio_len = audio_data.shape[0]
    desire_len = (self.audio_processor._config.voc_seq_frame +
                  self.audio_processor._config.pad *
                  2) * self.audio_processor.hop_length + (
                      self.audio_processor.win_length -
                      self.audio_processor.hop_length) + 1
    if audio_len <= desire_len:
      return np.pad(audio_data, [desire_len - audio_len, 0])
    start_id = np.random.randint(audio_len - desire_len + 1)
    end_id = start_id + desire_len
    return audio_data[start_id:end_id]

  def stft(self, audio_data):
    emphasised_data = tf.numpy_function(self.audio_processor.preemphasis,
                                        [audio_data], tf.float32)[..., 1:]
    audio_data = audio_data[..., 1:]
    audio_frames = tf.numpy_function(self.audio_processor.frames, [audio_data],
                                     tf.float32)

    audio_windowed = tf.numpy_function(
        lambda x: x * self.audio_processor.fft_window, [audio_frames],
        tf.float32)
    audio_frames_after_win_and_pad = tf.numpy_function(
        self.audio_processor.pad_center, [audio_windowed], tf.float32)
    return audio_frames_after_win_and_pad, audio_data, emphasised_data

  def input_and_label(self, audio_data):
    pad = self.audio_processor._config.pad * self.audio_processor.hop_length
    return audio_data[pad - 1:-pad - 1], audio_data[pad:-pad]

  def normalize_and_label(self, x, y, z):
    normalized = tf.numpy_function(self.audio_processor.normalize, [x],
                                   tf.float32)

    return normalized, self.input_and_label(y)


# TODO: add trim silence
def load_acoustic_dataset(config: dict) -> tf.data.Dataset:
  wav_paths = read_metadata(Path(config.metadata))
  loader = WaveRNNLoader(config.audio_process)
  dataset = tf.data.Dataset.from_tensor_slices(wav_paths).shuffle(256)
  dataset = dataset.map(lambda x: tf.numpy_function(
      loader.random_desired_samples, [x], tf.float32))
  dataset = dataset.batch(config.train.batch_size, drop_remainder=True)
  dataset = dataset.map(loader.stft)
  dataset = dataset.map(loader.normalize_and_label)
  return dataset.prefetch(100)
