import pandas as pd
import numpy as np


def load_annotations(filename,
                     n_top=50,
                     n_train_shards=16,
                     n_val_shards=2,
                     n_test_shards=6):
  """Reads annotation file, takes top N tags, and splits data samples.

  Results 54 (top50_tags + [clip_id, mp3_path, split, shard]) columns:

    ['choral', 'female voice', 'metal', 'country', 'weird', 'no voice',
     'cello', 'harp', 'beats', 'female vocal', 'male voice', 'dance',
     'new age', 'voice', 'choir', 'classic', 'man', 'solo', 'sitar', 'soft',
     'pop', 'no vocal', 'male vocal', 'woman', 'flute', 'quiet', 'loud',
     'harpsichord', 'no vocals', 'vocals', 'singing', 'male', 'opera',
     'indian', 'female', 'synth', 'vocal', 'violin', 'beat', 'ambient',
     'piano', 'fast', 'rock', 'electronic', 'drums', 'strings', 'techno',
     'slow', 'classical', 'guitar', 'clip_id', 'mp3_path', 'split', 'shard']
     
  NOTE: This will exclude audios which have only zero-tags. Therefore, number of
    each split will be 15250 / 1529 / 4332 (training / validation / test).

  Args:
    filename: A path to annotation CSV file.
    n_top: Number of the most popular tags to take.
    n_train_shards: Number of shards in training TFRecord files.
    n_val_shards: Number of shards in validation TFRecord files.
    n_test_shards: Number of shards in test TFRecord files.

  Returns:
    A DataFrame contains information of audios.

    Schema:
      <tags>: 0 or 1
      clip_id: clip_id of the original dataset
      mp3_path: A path to a mp3 audio file.
      split: A split of dataset (training / validation / test).
             The split is determined by its directory (0, 1, ... , f).
             First 12 directories (0 ~ b) are used for training,
             1 (c) for validation, and 3 (d ~ f) for test.
      shard: A shard index of the audio.
  """
  df = pd.read_csv(filename, delimiter='\t')

  top50 = (df.drop(['clip_id', 'mp3_path'], axis=1)
           .sum()
           .sort_values()
           .tail(n_top)
           .index
           .tolist())

  df = df[top50 + ['clip_id', 'mp3_path']]

  def split_by_directory(mp3_path):
    directory = mp3_path.split('/')[0]
    part = int(directory, 16)

    if part in range(12):
      return 'train'
    elif part is 12:
      return 'val'
    elif part in range(13, 16):
      return 'test'

  df['split'] = df['mp3_path'].apply(
    lambda mp3_path: split_by_directory(mp3_path))

  np.random.seed(123)

  for split, n_shards in [('train', n_train_shards),
                          ('val', n_val_shards),
                          ('test', n_test_shards)]:
    n_samples = sum(df['split'] == split)

    shards = np.random.randint(n_shards, size=n_samples)
    shards = shards[:n_samples]

    df.loc[df['split'] == split, 'shard'] = shards

  df['shard'] = df['shard'].astype(int)

  # Exclude rows which only have zeros.
  df = df.ix[~(df.ix[:, :n_top] == 0).all(axis=1)]

  return df
