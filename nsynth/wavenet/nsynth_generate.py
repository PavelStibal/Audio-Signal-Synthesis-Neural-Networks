# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A binary for generating samples given a folder of .wav files or encodings."""

import os

import librosa
import numpy as np
import soundfile
import tensorflow as tf
import matplotlib.pyplot as plt

from nsynth import utils
from nsynth.wavenet import fastgen

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("save_path", "", "Path to output file dir.")
tf.app.flags.DEFINE_string("checkpoint_path", "model.ckpt-200000", "Path to checkpoint.")
tf.app.flags.DEFINE_string("log", "INFO", "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")
tf.app.flags.DEFINE_integer("gpu_number", 0, "Number of the gpu to use for multigpu generation.")
tf.app.flags.DEFINE_string("file1_path", "", "Path to file_1 with either .wav.")
tf.app.flags.DEFINE_string("file2_path", "", "Path to file_2 with either .wav.")
tf.app.flags.DEFINE_integer("type", 0, "Combination of two audio signals. 0 - Combination by linear interpolation "
                                       "or 1 - Combination by crossfade audio.")
tf.app.flags.DEFINE_integer("num_type", 5, "Number for two audio signals. Number could be between 0 and 10. "
                                           "0 - Generated audio signal will be similar to the 1st audio signal, "
                                           "5 - Generated audio signal will be something between the 1st and 2nd "
                                           "audio signals 10 - Generated audio signal will be similar to the 2nd "
                                           "audio signal")
tf.app.flags.DEFINE_boolean("withoutNN", False, "Combination of two audio signals with NN and without NN")


def create_figure_audios(audio1, audio2, num_type, encoding1, encoding2):
    """figure audios used for development """

    fig, axs = plt.subplots(5, 1, figsize=(10, 7), sharex=True, sharey=True)
    axs[0].plot(audio1)
    axs[0].set_title('audio1')
    axs[1].plot(audio2)
    axs[1].set_title('audio2')
    axs[2].plot(linear_interpolation(audio1, audio2, num_type))
    axs[2].set_title('mix')
    axs[3].plot(crossfade_audio(audio1, audio2, num_type))
    axs[3].set_title('c1')
    axs[4].plot(crossfade_audio(audio2, audio1, num_type))
    axs[4].set_title('c2')
    plt.show()

    fig, axs = plt.subplots(5, 1, figsize=(10, 7), sharex=True, sharey=True)
    axs[0].plot(encoding1[0])
    axs[0].set_title('Encoding1')
    axs[1].plot(encoding2[0])
    axs[1].set_title('Encoding2')
    axs[2].plot(linear_interpolation(encoding1, encoding2, num_type)[0])
    axs[2].set_title('mix')
    axs[3].plot(crossfade(encoding2, encoding1, num_type)[0])
    axs[3].set_title('Crossfade')
    axs[4].plot(crossfade(encoding1, encoding2, num_type)[0])
    axs[4].set_title('Crossfade')
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(encoding1[0])
    axs[0].set_title('Original Encoding')
    axs[1].plot(fade(encoding1, 'in')[0])
    axs[1].set_title('Fade In')
    axs[2].plot(fade(encoding1, 'out')[0])
    axs[2].set_title('Fade Out')
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(encoding2[0])
    axs[0].set_title('Original Encoding')
    axs[1].plot(fade(encoding2, 'in')[0])
    axs[1].set_title('Fade In')
    axs[2].plot(fade(encoding2, 'out')[0])
    axs[2].set_title('Fade Out')
    plt.show()


def load_audio(fname):
    return soundfile.read(fname)


def encoding_audio(audio, ckpt='model.ckpt-200000'):
    return fastgen.encode(audio, ckpt, len(audio))


def create_same_length(len_num1, len_num2, data1, data2):
    """create the same size with zero

        Args:
            len_num1: size of 1st data
            len_num2: size of 2nd data
            data1: 1st data
            data2: 2nd data

        Returns:
            data1, data2: same size datas.
    """
    if len_num1 < len_num2:
        data1 = np.hstack([data1, np.zeros(len_num2 - len_num1)])
    elif len_num1 > len_num2:
        data2 = np.hstack([data2, np.zeros(len_num1 - len_num2)])

    return data1, data2


def resampling(data, required_samplerate, samplerate):
    """Resampling to the desired frequency

        Args:
            data: data from sound
            required_samplerate: required sample rate
            samplerate: original sample rate

        Returns:
            new_data: resampling data.
    """
    return librosa.resample(data, samplerate, required_samplerate)


def create_same_sample_rate(data1, sample_rate1, data2, sample_rate2):
    """create the same sample rate for both audio

        Args:
            data1: 1st data
            sample_rate1: sample rate for 1st data
            data2: 2nd data
            sample_rate2: sample rate for 2nd data

        Returns:
            data1, sample_rate1, data2, sample_rate2: data with same sample rate.
    """

    if sample_rate1 < sample_rate2:
        data1 = resampling(data1, sample_rate2, sample_rate1)
        sample_rate1 = sample_rate2
    elif sample_rate1 > sample_rate2:
        data2 = resampling(data2, sample_rate1, sample_rate2)
        sample_rate2 = sample_rate1

    return data1, sample_rate1, data2, sample_rate2


def get_num_for_intepolation(x):
    """Get number for linear interpolation
       Is used to change amplitude audios
    """
    return {
        1: 0.2,
        2: 0.4,
        3: 0.6,
        4: 0.8,
        5: 1,
        6: 0.8,
        7: 0.6,
        8: 0.4,
        9: 0.2,
    }[x]


def get_num_for_crossfade(x):
    """Get number for  crossfade
       Is used to change fade audios
    """
    return {
        1: 0,
        2: 0.1125,
        3: 0.225,
        4: 0.3575,
        5: 0.45,
        6: 0.65,
        7: 0.7,
        8: 0.85,
        9: 1,
    }[x]


def linear_interpolation(data1, data2, type_num):
    """create linear interpolation for data

        Args:
            data1: 1st data
            data2: 2nd data
            type_num: how much the signals will hear?

        Returns:
            data: combination data1 and data2.
    """

    if type_num == 0:
        return data1
    elif type_num == 10:
        return data2
    elif 0 < type_num < 5:
        data2 *= get_num_for_intepolation(type_num)
        data1 *= (1.0 - get_num_for_intepolation(type_num))
    elif 5 < type_num < 10:
        data1 *= get_num_for_intepolation(type_num)
        data2 *= (1.0 - get_num_for_intepolation(type_num))

    return (data1 + data2) / 2


def fade(encoding, type_num, mode='in'):
    """create fade in of fade out for crossfade between two encoding audios """

    length = encoding.shape[1]
    start_num = get_num_for_crossfade(type_num)
    fadein = np.linspace(start_num, (1 - start_num), length).reshape(1, -1, 1)

    if mode == 'in':
        return fadein * encoding
    else:
        return (1.0 - fadein) * encoding


def fade_audio(audio, type_num, mode='in'):
    """create fade in of fade out for crossfade between two audios """
    length = len(audio)
    start_num = get_num_for_crossfade(type_num)
    fadein = np.linspace(start_num, (1 - start_num), length)

    if mode == 'in':
        return fadein * audio
    else:
        return (1.0 - fadein) * audio


def crossfade_audio(audio1, audio2, type_num):
    """create crossfade for audios

        Args:
            audio1: 1st data
            audio2: 2nd data
            type_num: how much the signals will be overlapped?

        Returns:
            combination audio1 and audio2.
    """
    if type_num == 0:
        return audio1
    elif type_num == 10:
        return audio2

    return fade_audio(audio1, type_num, 'out') + fade_audio(audio2, type_num, 'in')


def crossfade(encoding1, encoding2, type_num):
    """create crossfade for encoding audios

        Args:
            encoding1: 1st data
            encoding2: 2nd data
            type_num: how much the signals will be overlapped?

        Returns:
            combination encoding1 and encoding2.
    """
    if type_num == 0:
        return encoding1
    elif type_num == 10:
        return encoding2

    return fade(encoding1, type_num, 'out') + fade(encoding2, type_num, 'in')


def check_is_exists_file(filename):
    """Check is exists file

        Args:
            filename: path for file..

        Returns:
            bool: true if exists, false if not exists.
    """
    return os.path.isfile(filename)


def fill_slash_in_output_path(output_path):
    """Check output path whether it ends with a slash
       if not, slash is filled
    """
    if not output_path.endswith('/'):
        output_path += '/'

    return output_path


def check_input_parametres(file1_path, file2_path, type_mix, num_type):
    """Check inputs"""

    if file1_path == "" or file2_path == "":
        raise Exception('No entered file1_path or file2_path')
    elif not check_is_exists_file(file1_path) or not check_is_exists_file(file2_path):
        raise Exception('Incorrectly entered two audio file')

    if not file1_path.endswith('.wav') or not file2_path.endswith('.wav'):
        raise RuntimeError("Files must be .wav")

    if 0 < type_mix and type_mix > 1:
        raise Exception('Incorrectly entered number of type')

    if 0 < num_type and num_type > 10:
        raise Exception('Incorrectly entered number of synth')


def main(unused_argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_number)

    file1_path = FLAGS.file1_path
    file2_path = FLAGS.file2_path
    type_mix = FLAGS.type
    num_type = FLAGS.num_type
    check_input_parametres(file1_path, file2_path, type_mix, num_type)

    checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
    save_path = utils.shell_path(FLAGS.save_path)

    audio1, sample_rate1 = load_audio(file1_path)
    audio2, sample_rate2 = load_audio(file2_path)

    print("load audio")

    audio1, sample_rate1, audio2, sample_rate2 = create_same_sample_rate(audio1, sample_rate1, audio2, sample_rate2)

    audio1, audio2 = create_same_length(len(audio1), len(audio2), audio1, audio2)

    print("same len")

    encoding1 = encoding_audio(audio1, checkpoint_path)
    encoding2 = encoding_audio(audio2, checkpoint_path)

    print("encoding done")

    # save synthesis of sound
    save_name = fill_slash_in_output_path(save_path)
    if FLAGS.withoutNN:
        if type_mix == 0:
            audio = linear_interpolation(audio1, audio2, num_type)
            save_name += 'mix.wav'
        else:
            audio = crossfade_audio(audio1, audio2, num_type)
            save_name += 'crossfade.wav'

        soundfile.write(save_name, audio, sample_rate1)
        print("save audio")

    # save synthesis of encoding sound
    save_name = fill_slash_in_output_path(save_path)
    if type_mix == 0:
        encodings = linear_interpolation(encoding1, encoding2, num_type)
        save_name += 'mixNN.wav'
    else:
        encodings = crossfade(encoding1, encoding2, num_type)
        save_name = 'crossfadeNN.wav'

    if FLAGS.gpu_number != 0:
        with tf.device("/device:GPU:%d" % FLAGS.gpu_number):
            fastgen.synthesize(encodings, save_paths=[save_name], checkpoint_path=checkpoint_path)
    else:
        fastgen.synthesize(encodings, save_paths=[save_name], checkpoint_path=checkpoint_path)

    print("save audio, which was generate by NN")


def console_entry_point():
    tf.app.run(main)


if __name__ == "__main__":
    console_entry_point()
