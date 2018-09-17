import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import tensorflow as tf

from preprocesor_autoencoder import utils
from preprocesor_autoencoder.wavenet import fastgen

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("mode", 1, "Type of mode. 0 - load two audio files "
                                       "1 - randomized load multiple audio files")

tf.app.flags.DEFINE_string("input_path", "", "Path to directory with either .wav files.")
tf.app.flags.DEFINE_string("output_path", "", "Path to output file dir.")
tf.app.flags.DEFINE_string("checkpoint_path", "model.ckpt-200000", "Path to checkpoint.")

tf.app.flags.DEFINE_integer("batch_size", 1, "Number of samples per a batch.")
tf.app.flags.DEFINE_integer("gpu_number", 0, "Number of the gpu to use for multigpu generation.")

tf.app.flags.DEFINE_string("file1_path", "", "Path to file_1 with either .wav.")
tf.app.flags.DEFINE_integer("file1_amp", 1, "Amplitude of audio signal. 1 = original.")
tf.app.flags.DEFINE_integer("file1_f", 0, "Frequency of audio signal.")
tf.app.flags.DEFINE_string("file2_path", "", "Path to file_2 with either .wav.")
tf.app.flags.DEFINE_integer("file2_amp", 1, "Amplitude of audio signal. 1 = original.")
tf.app.flags.DEFINE_integer("file2_f", 0, "Frequency of audio signal.")

tf.app.flags.DEFINE_integer("type", 0, "Combination of two audio signals. "
                                       "0 - Combination by additive synthesis "
                                       "or 1 - Combination by crossfade audio.")
tf.app.flags.DEFINE_integer("num_type", 5, "Number for two audio signals. Number could be between 0 and 10. "
                                           "0 - Generated audio signal will be similar to the 1st audio signal, "
                                           "5 - Generated audio signal will be something between the 1st and 2nd "
                                           "audio signals 10 - Generated audio signal will be similar to the 2nd "
                                           "audio signal")


def create_figure_all_audio_data(original1_data, original2_data, together_data):
    """figure all adudios.

      Args:
        original1_data: data from 1st original sound
        original1_data: data from 2nd original sound
        together_data: data from combination sound

      """
    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(original1_data, '-r')
    plt.title('Encode audio:')

    plt.subplot(3, 1, 2)
    plt.plot(original2_data, '-b')
    plt.title('Decode audio:')

    plt.subplot(3, 1, 3)
    plt.plot(together_data, '-g')
    plt.title('Original audio:')

    plt.savefig('signals1.png')


def get_num_for_synth(x):
    """Get number for .additive synthesis"""
    return {
        0: 0,
        1: 0.2,
        2: 0.4,
        3: 0.6,
        4: 0.8,
        5: 1,
        6: 0.8,
        7: 0.6,
        8: 0.4,
        9: 0.2,
        10: 0
    }[x]


def get_num_for_cross(x):
    """Get number for .length crossfade"""
    return {
        0: 0.05,
        1: 0.15,
        2: 0.25,
        3: 0.35,
        4: 0.45,
        5: 0.55,
        6: 0.65,
        7: 0.75,
        8: 0.85,
        9: 0.95,
        10: 1
    }[x]


def check_is_exists_file(filename):
    """Check is exists file

        Args:
            filename: path for file..

        Returns:
            bool: true if exists, false if not exists.
    """
    return os.path.isfile(filename)


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


def additive_synthesis(data1, data2, type_num):
    """create additive synthesis

        Args:
            data1: 1st data
            data2: 2nd data
            type_num: how much the signals will hear?

        Returns:
            data: combination data1 and data2.
    """
    data1, data2 = create_same_length(len(data1), len(data2), data1, data2)

    if -1 < type_num < 5:
        data2 *= get_num_for_synth(type_num)
    elif 5 < type_num < 11:
        data1 *= get_num_for_synth(type_num)

    return (data1 + data2) / 2


def calculate_crossfade_length(len_num1, len_num2, type_num):
    """Calculate length of overlap

        Args:
            len_num1: size of 1st data
            len_num2: size of 2nd data
            type_num: how much the signals will be overlapped?

        Returns:
            data: combination data1 and data2.
    """
    lenght = 0

    if len_num1 < len_num2:
        lenght = len_num1
    else:
        lenght = len_num2

    return int(lenght * get_num_for_cross(type_num))


def crossfade_audio(data1, data2, type_num):
    """create additive synthesis

        Args:
            data1: 1st data
            data2: 2nd data
            type_num: how much the signals will be overlapped?

        Returns:
            data: combination data1 and data2.
    """
    crossfade = calculate_crossfade_length(len(data1), len(data2), type_num)
    data = np.zeros(len(data1) + len(data2) - crossfade)
    data[:len(data1)] = data1

    modulo = crossfade / 4
    facein = 0.25
    faceout = 0.75

    for i in range(0, len(data2)):
        if 0 != i and (i % modulo) == 0 and i < crossfade:
            facein += 0.25
            faceout -= 0.25

        data[len(data1) + i - crossfade] *= faceout
        data[len(data1) + i - crossfade] += data2[i] * facein

    return data


def save_audio(data, sample_rate, path):
    """Save audio file (wav)"""
    soundfile.write(path, data, sample_rate)


def preprocesor_audio(file1_path, file2_path, mode, type_mix, num_type):
    """preparing everything necessary before using neural networks

        Args:
            file1_path: 1st file for create data1 and sample_rate1
            file2_path: 2nd file for create data2 and sample_rate2
            mode: processing only two files or multiple files
            type_mix: additive synthesis or crossfade
            num_type: how much the signals will be overlapped? or how much the signals will hear?

        Returns:
            data: combination data1 and data2.
    """
    name = str(type_mix) + "_" + os.path.splitext(os.path.basename(file1_path))[0] + "_" + \
           os.path.splitext(os.path.basename(file2_path))[0] + "_" + str(num_type) + ".wav"

    data1, sample_rate1 = soundfile.read(file1_path)
    data2, sample_rate2 = soundfile.read(file2_path)
    sample_rate = sample_rate1
    data = []

    if mode == 0:
        data1 *= FLAGS.file1_amp
        data2 *= FLAGS.file2_amp

        if FLAGS.file1_f != 0:
            data1 = resampling(data1, FLAGS.file1_f, sample_rate1)
            sample_rate = FLAGS.file1_f

        if FLAGS.file2_f != 0:
            data2 = resampling(data2, FLAGS.file2_f, sample_rate2)
            sample_rate = FLAGS.file2_f

    if sample_rate1 < sample_rate2:
        data1 = resampling(data1, sample_rate2, sample_rate1)
        sample_rate = sample_rate2
    elif sample_rate1 > sample_rate2:
        data2 = resampling(data2, sample_rate1, sample_rate2)
        sample_rate = sample_rate2

    if type_mix == 0:
        data = additive_synthesis(data1, data2, num_type)
    elif type_mix == 1:
        data = crossfade_audio(data1, data2, num_type)

    return name, data, sample_rate


def check_input_parametres(mode, file1_path, file2_path, input_path, type_mix, num_type):
    """Check inputs"""

    if mode == 0:
        if file1_path == "" and file2_path == "":
            raise Exception('No entered file1_path or file2_path')
        elif not check_is_exists_file(file1_path) and not check_is_exists_file(file2_path):
            raise Exception('Incorrectly entered two audio file')
    elif mode == 1:
        if input_path == "":
            raise Exception('No entered input_path')
        elif not tf.gfile.IsDirectory(input_path):
            raise Exception('Incorrectly entered input directory')
    else:
        raise Exception('Incorrectly entered number of mode')

    if 0 < type_mix and type_mix > 1:
        raise Exception('Incorrectly entered number of type')

    if 0 < num_type and num_type > 10:
        raise Exception('Incorrectly entered number of type')


def main(unused_argv=None):
    """Main of program"""
    tf.logging.set_verbosity("INFO")

    mode = FLAGS.mode
    file1_path = FLAGS.file1_path
    file2_path = FLAGS.file2_path
    input_path = FLAGS.input_path
    type_mix = FLAGS.type
    num_type = FLAGS.num_type
    check_input_parametres(mode, file1_path, file2_path, input_path, type_mix, num_type)

    # preparing everything necessary before using neural networks
    datas = []
    sample_rates = []
    names = []
    if mode == 0:
        tf.logging.info("Make preprocessor for 2 sounds")
        name, data, sample_rate = preprocesor_audio(file1_path, file2_path, mode, type_mix, num_type)
        names.append(name)
        datas.append(data)
        sample_rates.append(sample_rate)
    else:
        input_path = utils.shell_path(FLAGS.input_path)

        files = tf.gfile.ListDirectory(input_path)
        exts = [os.path.splitext(f)[1] for f in files]

        if ".wav" in exts:
            postfix = ".wav"
        else:
            raise RuntimeError("Folder must contain .wav")

        files = ([
            os.path.join(input_path, fname)
            for fname in files
            if fname.lower().endswith(postfix)
        ])

        tf.logging.info("Make preprocessor for " + str(len(files) - 1) + " sounds")
        # k = 0
        # for i in range(0, len(files) - 2, 2):
        #     for j in range(0, 11):
        #         name, data, sample_rate = preprocesor_audio(files[i], files[i + 1], mode, type_mix, j)
        #         names.append(name)
        #         datas.append(data)
        #         sample_rates.append(sample_rate)
        #
        #     k += 11
        #     tf.logging.info(str(k) + " is ready combination")

        for i in range(0, len(files) - 1):
            name = os.path.splitext(os.path.basename(files[i]))[0] + ".wav"

            data, sample_rate = soundfile.read(files[i])
            names.append(name)
            datas.append(data)
            sample_rates.append(sample_rate)

            tf.logging.info(str(i + 1) + " is ready sounds")

    # after preparing
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_number)
    checkpoint_path = utils.shell_path(FLAGS.checkpoint_path)
    output_path = utils.shell_path(FLAGS.output_path)

    if not output_path:
        raise RuntimeError("Must specify output_path.")

    # Now synthesize from files one batch at a time
    batch_size = FLAGS.batch_size
    n = len(datas)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_data = datas[start:end]
        lenght = len(batch_data[0])
        name = names[start:end]
        sample_rate = sample_rates[start:end]

        # save_audio(batch_data[0], sample_rate[0], os.path.join(output_path, name[0]))

        batch_data = fastgen.load_batch(batch_data)

        encodings = fastgen.encode(batch_data, checkpoint_path, lenght)

        if FLAGS.gpu_number != 0:
            with tf.device("/device:GPU:%d" % FLAGS.gpu_number):
                fastgen.synthesize(
                    encodings, os.path.join(output_path, "gen_" + name[0]), sample_rate[0],
                    checkpoint_path=checkpoint_path)
        else:
            fastgen.synthesize(encodings, os.path.join(output_path, "gen_" + name[0]),
                               sample_rate[0], checkpoint_path=checkpoint_path)


def console_entry_point():
    tf.app.run(main)


if __name__ == "__main__":
    console_entry_point()
