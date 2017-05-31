"""Converts MagnaTagATune dataset to TFRecord files. """

import os

import pandas as pd
import tensorflow as tf

from datetime import datetime
from multiprocessing import Pool

from madmom.audio.signal import LoadAudioFileError

from sample_cnn.data.audio_processing import (audio_to_sequence_example,
                                              audio_to_examples)
from sample_cnn.data.mtt.annotation_processing import load_annotations

tf.flags.DEFINE_string('data_dir', '', 'MagnaTagATune audio dataset directory.')
tf.flags.DEFINE_string('annotation_file', '',
                       'A path to CSV annotation file which contains labels.')
tf.flags.DEFINE_string('output_dir', '', 'Output data directory.')
tf.flags.DEFINE_string('output_labels', '', 'Output label file.')

tf.flags.DEFINE_integer('n_top', 50, 'Number of top N tags.')

tf.flags.DEFINE_integer('n_processes', 4,
                        'Number of processes to process audios in parallel.')

tf.flags.DEFINE_integer('n_audios_per_shard', 100,
                        'Number of audios per shard.')

# Audio processing flags
tf.flags.DEFINE_integer('sample_rate', 22050, 'Sample rate of audio.')
tf.flags.DEFINE_integer('n_samples', 59049, 'Number of samples per segment.')

FLAGS = tf.flags.FLAGS


def _process_audio_files(process_idx, assigned_anno, sample_rate, n_samples,
                         split, shard, n_shards):
  """Processes and saves audios as TFRecord files in one sub-process.
  
  Args:
    process_idx: Integer process identifier.
    assigned_anno: A DataFrame which contains information about the audios 
      that should be process in this sub-process.
    sample_rate: Sampling rate of the audios. If the sampling rate is different 
      with an audio's original sampling rate, then it re-samples the audio.
    n_samples: Number of samples one segment contains.
    split: Dataset split which is one of 'train', 'val', or 'test'.
    shard: Shard index.
    n_shards: Number of the entire shards.
  """
  is_train = (split == 'train')

  output_filename_format = ('{}-{:03d}-of-{:03d}.tfrecords'
                            if is_train else
                            '{}-{:03d}-of-{:03d}.seq.tfrecords')

  output_filename = output_filename_format.format(split, shard, n_shards)
  output_file_path = os.path.join(FLAGS.output_dir, output_filename)
  writer = tf.python_io.TFRecordWriter(output_file_path)

  for _, row in assigned_anno.iterrows():
    audio_path = os.path.join(FLAGS.data_dir, row['mp3_path'])
    labels = row[:FLAGS.n_top].tolist()

    try:
      if is_train:
        examples = audio_to_examples(audio_path, labels, sample_rate, n_samples)
      else:
        examples = [audio_to_sequence_example(audio_path, labels,
                                              sample_rate, n_samples)]
    except LoadAudioFileError:
      # There are some broken mp3 files. Ignore it.
      print('Cannot load audio "{}". Ignore it.'.format(audio_path))
      continue

    for example in examples:
      writer.write(example.SerializeToString())

  writer.close()

  print('{} [process {:02d}]: Done processing {} songs.'
        .format(datetime.now(), process_idx, len(assigned_anno)))


def _process_dataset(anno, sample_rate, n_samples, n_processes):
  """Processes, and saves MagnaTagATune dataset using multi-processes.

  Args:
    anno: Annotation DataFrame contains tags, mp3_path, split, and shard.
    sample_rate: Sampling rate of the audios. If the sampling rate is different 
      with an audio's original sampling rate, then it re-samples the audio.
    n_samples: Number of samples one segment contains.
    n_processes: Number of processes to process the dataset.
  """
  args_for_processes = []
  split_and_shard_sets = pd.unique(anno[['split', 'shard']].values)

  for process_idx, (split, shard) in enumerate(split_and_shard_sets):
    assigned_anno = anno[(anno['split'] == split) & (anno['shard'] == shard)]
    n_shards = anno[anno['split'] == split]['shard'].nunique()

    args = (process_idx, assigned_anno, sample_rate, n_samples,
            split, shard, n_shards)
    args_for_processes.append(args)

  if FLAGS.n_processes > 1:
    # For each split and shard set, create process
    with Pool(processes=n_processes) as pool:
      pool.starmap(_process_audio_files, args_for_processes)
  else:
    _process_audio_files(*args_for_processes[0])


def _save_tags(tag_list, output_labels):
  """Saves a list of tags to a file.
  
  Args:
    tag_list: The list of tags.
    output_labels: A path to save the list.
  """
  with open(output_labels, 'w') as f:
    f.write('\n'.join(tag_list))


def main(unused_argv):
  df = load_annotations(filename=FLAGS.annotation_file,
                        n_top=FLAGS.n_top,
                        n_audios_per_shard=FLAGS.n_audios_per_shard)

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.logging.info('Creating output directory: %s', FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Save top N tags
  tag_list = df.columns[:FLAGS.n_top].tolist()
  _save_tags(tag_list, FLAGS.output_labels)
  print('Top {} tags written to {}'.format(len(tag_list), FLAGS.output_labels))

  df_train = df[df['split'] == 'train']
  df_val = df[df['split'] == 'val']
  df_test = df[df['split'] == 'test']

  n_train = len(df_train)
  n_val = len(df_val)
  n_test = len(df_test)
  print('Number of songs for each split: {} / {} / {} '
        '(training / validation / test)'.format(n_train, n_val, n_test))

  n_train_shards = df_train['shard'].nunique()
  n_val_shards = df_val['shard'].nunique()
  n_test_shards = df_test['shard'].nunique()
  print('Number of shards for each split: {} / {} / {} '
        '(training / validation / test)'.format(n_train_shards,
                                                n_val_shards, n_test_shards))

  print('Start processing MagnaTagATune using {} cores'
        .format(FLAGS.n_processes))
  _process_dataset(df, FLAGS.sample_rate, FLAGS.n_samples, FLAGS.n_processes)

  print()
  print('Done.')


if __name__ == '__main__':
  tf.app.run()
