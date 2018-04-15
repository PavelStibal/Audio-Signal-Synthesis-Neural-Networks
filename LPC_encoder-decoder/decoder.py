import numpy as np
from scipy.signal import lfilter


def lpc_synth(excitation, a, lpc_mem):
    coeff = np.zeros(len(a) + 1)
    coeff[0:len(a)] = a

    return lfilter(coeff, 1, excitation, -1, lpc_mem)


def init_lpc_decoder(frame_length, nframes, lpcOrder):
    decode_data = np.zeros(np.dot(int(nframes), int(frame_length)))

    lpc_mem = np.zeros(lpcOrder)
    idx = np.arange(frame_length)

    return decode_data, lpc_mem, idx


def lpc_decoder(frame_length, nframes, lpcOrder, coeff, residual, amp):
    decode_data, lpc_mem, idx = init_lpc_decoder(frame_length, nframes, lpcOrder)

    for i in np.arange(0, nframes).reshape(-1):
        a = coeff[i, :]
        excitation = residual[i, :]

        lpc_output, lpc_mem = lpc_synth(excitation, a, lpc_mem)

        lpc_power = np.sqrt(np.dot(lpc_output.T, lpc_output)) / frame_length
        if lpc_power > 0:
            gain = amp[i] / lpc_power
        else:
            gain = 0.0

        frames = np.dot(gain, lpc_output)

        # re-construct the output signal
        decode_data[idx] = frames
        idx += frame_length

    decode_data = decode_data / 2 ** 15
    return decode_data
