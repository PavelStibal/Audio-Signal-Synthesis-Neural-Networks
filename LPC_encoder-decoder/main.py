import argparse
import os
import wave

import matplotlib.pyplot as plt
import numpy as np
import soundfile

import decoder as de
import encoder as en

FILE = 'wavfile.wav'
OUTPUT_FILE = 'wavfile-1.wav'
TEST_OUTPUT_FILE = 'wavfile-test.wav'
FRAMES = 0


def get_arguments():
    parser = argparse.ArgumentParser(description='LPC encoder and decoder')
    parser.add_argument('-filename', '--filename', dest='filename', type=str, default=FILE,
                        help='Path to file, which to be processed. Default: ' + FILE + '.')
    parser.add_argument('-frame', '--frame', dest='frame_length', type=int, default=FRAMES,
                        help='How long is frame? Default: ' + str(FRAMES) + '.')
    parser.add_argument('-out', '--out', dest='output_filename', type=str, default=OUTPUT_FILE,
                        help='Name of output file. Default: ' + OUTPUT_FILE + '.')

    return parser.parse_args()


def check_is_exists_file(filename):
    return os.path.isfile(filename)


def get_data_from_file(filename):
    data, samplerate = soundfile.read(filename)
    wave_data = wave.open(filename, 'r')

    return data, samplerate, wave_data


def calculate_frame_length(frame_length, data_length, nframes):
    if data_length != nframes:
        frame_length = int(np.round(data_length / nframes))
        nframes = int(np.floor(data_length / frame_length))

        return frame_length, nframes

    if -1 < frame_length < 2:
        if data_length > nframes:
            frame_length = int(np.round(data_length / nframes))
        else:
            frame_length = 2
            nframes = int(np.floor(data_length / frame_length))

        return frame_length, nframes

    nframes = int(np.floor(data_length / frame_length))

    return frame_length, nframes


def create_figure_all_audio_data_together(original_data, decode_data, encode_data):
    plt.figure(1)

    plt.plot(encode_data, '-r')
    plt.plot(decode_data, '-b')
    plt.plot(original_data, '-g')
    plt.title('Audios together:')

    plt.savefig('signals1.png')


def create_figure_all_audio_data(original_data, decode_data, encode_data):
    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(encode_data, '-r')
    plt.title('Encode audio:')

    plt.subplot(3, 1, 2)
    plt.plot(decode_data, '-b')
    plt.title('Decode audio:')

    plt.subplot(3, 1, 3)
    plt.plot(original_data, '-g')
    plt.title('Original audio:')

    plt.savefig('signals2.png')


def create_setrogram_all_audio_data(original_data, decode_data, encode_data, samplerate):
    plt.figure(3)
    plt.subplot(3, 1, 1)
    plt.specgram(encode_data, Fs=samplerate)
    plt.title('Encode audio:')

    plt.subplot(3, 1, 2)
    plt.specgram(decode_data, Fs=samplerate)
    plt.title('Decode audio:')

    plt.subplot(3, 1, 3)
    plt.specgram(original_data, Fs=samplerate)
    plt.title('Original audio:')

    plt.savefig('spectrograms.png')


def create_figures(original_data, decode_data, encode_data, samplerate):
    # create 1st figure
    create_figure_all_audio_data_together(original_data, decode_data, encode_data)
    # create 2nd figure
    create_figure_all_audio_data(original_data, decode_data, encode_data)
    # create 3rd figure
    create_setrogram_all_audio_data(original_data, decode_data, encode_data, samplerate)


if __name__ == '__main__':
    args = get_arguments()

    if not check_is_exists_file(args.filename):
        raise Exception('Incorrectly entered path to file')

    print('Read audio data')
    original_data, samplerate, wave_data = get_data_from_file(args.filename)
    frame_length, nframes = calculate_frame_length(args.frame_length, len(original_data), wave_data.getnframes())

    print('Encode audio data')
    lpcOrder, amp, residual, aCoeff, encode_data = en.lpc_encoder(frame_length, nframes, original_data)

    print('Decode audio data')
    decode_data = de.lpc_decoder(frame_length, nframes, lpcOrder, aCoeff, residual, amp)

    print('Create figure')
    create_figures(original_data, decode_data, encode_data, samplerate)

    print('Create decode audio file')
    soundfile.write(args.output_filename, decode_data, samplerate)
