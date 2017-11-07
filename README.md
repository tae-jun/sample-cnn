# Sample CNN
***A TensorFlow implementation of "Sample-level Deep Convolutional
Neural Networks for Music Auto-tagging Using Raw Waveforms"***

This is a [TensorFlow][1] implementation of "[*Sample-level Deep
Convolutional Neural Networks for Music Auto-tagging Using Raw
Waveforms*][10]" using [Keras][11]. This repository only implements the
best model of the paper. (the model described in Table 1; m=3, n=9)


## Table of contents
* [Prerequisites](#prerequisites)
* [Preparing MagnaTagATune (MTT) Dataset](#preparing-mtt)
* [Preprocessing the MTT dataset](#preprocessing)
* [Training a model from scratch](#training)
* [Evaluating a model](#evaluating)


<a name="prerequisites"></a>
## Prerequisites
* Python 3.5 and the required packages
* `ffmpeg` (required for `madmom`)

### Installing required Python packages
```sh
pip install -r requirements.txt
pip install madmom
```
The `madmom` package has a install-time dependency, so should be
installed after installing packages in `requirements.txt`.

This will install the required packages:
* [tensorflow][1] **1.0.1** (has an issue on 1.1.0)
* [keras][11]
* [pandas][2]
* [scikit-learn][3]
* [madmom][4]
* [numpy][5]
* [scipy][6]
* [cython][7]
* [h5py][12]

### Installing ffmpeg
`ffmpeg` is required for `madmom`.

#### MacOS (with Homebrew):
```sh
brew install ffmpeg
```

#### Ubuntu:
```sh
add-apt-repository ppa:mc3man/trusty-media
apt-get update
apt-get dist-upgrade
apt-get install ffmpeg
```

#### CentOS:
```sh
yum install epel-release
rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
rpm -Uvh http://li.nux.ro/download/nux/dextop/el ... noarch.rpm
yum install ffmpeg
```


<a name="preparing-mtt"></a>
## Preparing MagnaTagATune (MTT) dataset
Download audio data and tag annotations from [here][8]. Then you should
see 3 `.zip` files and 1 `.csv` file:
```sh
mp3.zip.001
mp3.zip.002
mp3.zip.003
annotations_final.csv
```

To unzip the `.zip` files, merge and unzip them (referenced [here][9]):
 ```sh
 cat mp3.zip.* > mp3_all.zip
 unzip mp3_all.zip
 ```

You should see 16 directories named `0` to `f`. Typically, `0 ~ b` are
used to training, `c` to validation, and `d ~ f` to test.

To make your life easier, place them in a directory as below:
```sh
├── annotations_final.csv
└── raw
    ├── 0
    ├── 1
    ├── ...
    └── f
```

And we will call the directory `BASE_DIR`. Preparing the MTT dataset is Done!


<a name="preprocessing"></a>
## Preprocessing the MTT dataset
This section describes a required preprocessing task for the MTT
dataset. Note that this requires `57G` storage space.

These are what the preprocessing does:
* Select top 50 tags in `annotations_final.csv`
* Split dataset into training, validation, and test sets
* Segment the raw audio files into `59049` sample length
* Convert to TFRecord format

To run the preprocessing, copy a shell template and edit the copy:
```sh
cp scripts/build_mtt.sh.template scripts/build_mtt.sh
vi scripts/build_mtt.sh
```

You should fill in the environment variables:
* `BASE_DIR` the directory contains `annotations_final.csv` file and
  `raw` directory
* `N_PROCESSES` number of processes to use; the preprocessing uses
  multi-processing
* `ENV_NAME` (optional) if you use `virtualenv` or `conda` to create a
  separated environment, write your environment name

The below is an example:
```sh
BASE_DIR="/path/to/mtt/basedir"
N_PROCESSES=4
ENV_NAME="sample_cnn"
```

And run it:
```sh
./scripts/build_mtt.sh
```

The script will **automatically run a process in the background**, and
**tail output** which the process prints. This will take a few minutes
to an hour according to your device.

The converted TFRecord files will be located in your
`${BASE_DIR}/tfrecord`. Now, your `BASE_DIR`'s structure should be like
this:
```sh
├── annotations_final.csv
├── build_mtt.log
├── labels.txt
├── raw
│   ├── 0
│   ├── ...
│   └── f
└── tfrecord
    ├── test-000-of-036.seq.tfrecords
    ├── ...
    ├── test-035-of-036.seq.tfrecords
    ├── train-000-of-128.tfrecords
    ├── ...
    ├── train-127-of-128.tfrecords
    ├── val-000-of-012.seq.tfrecords
    ├── ...
    └── val-011-of-012.seq.tfrecords
```


<a name="training"></a>
## Training a model from scratch
To train a model from scratch, copy a shell template and edit the
copy like what did above:
```sh
cp scripts/train.sh.template scripts/train.sh
vi scripts/train.sh
```

And fill in the environment variables:
* `BASE_DIR` the directory contains `tfrecord` directory
* `TRAIN_DIR` where to save your trained model, and summaries to
  visualize your training using TensorBoard
* `ENV_NAME` (optional) if you use `virtualenv` or `conda` to create a
  separated environment, write your environment name

The below is an example:
```sh
BASE_DIR="/path/to/mtt/basedir"
TRAIN_DIR="/path/to/save/outputs"
ENV_NAME="sample_cnn"
```

Let's kick off the training!:
```sh
./scripts/train.sh
```

The script will **automatically run a process in the background**, and
**tail output** which the process prints.


<a name="evaluating"></a>
## Evaluating a model
Copy an evaluating shell script template and edit the copy:
```sh
cp scripts/evaluate.sh.template scripts/evaluate.sh
vi scripts/evaluate.sh
```

Fill in the environment variables:
* `BASE_DIR` the directory contains `tfrecord` directory
* `CHECKPOINT_DIR` where you saved your model (`TRAIN_DIR` when training)
* `ENV_NAME` (optional) if you use `virtualenv` or `conda` to create a
  separated environment, write your environment name

The script doesn't evaluate the latest model but the best model. If you
want to evaluate the latest model, you should give `--best=False` as an
option.

[1]: https://www.tensorflow.org/
[2]: http://pandas.pydata.org/
[3]: https://www.scipy.org/
[4]: https://madmom.readthedocs.io/en/latest/
[5]: http://www.numpy.org/
[6]: https://www.scipy.org
[7]: http://cython.org/
[8]: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
[9]: https://github.com/keunwoochoi/magnatagatune-list
[10]: https://arxiv.org/abs/1703.01789
[11]: https://keras.io/
[12]: http://www.h5py.org
