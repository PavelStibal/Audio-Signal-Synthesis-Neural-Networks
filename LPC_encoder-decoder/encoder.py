import numpy as np
from scipy.signal import lfilter
from scikits.talkbox import lpc

LPC_ORDER = 10


def calculate_lpc_order(frame_length):
    if LPC_ORDER > frame_length:
        return frame_length

    return LPC_ORDER


def lpc_gain(x):
    return np.sqrt(np.dot(x.T, x)) / len(x)


def lpc_filter(x, lpc_mem):
    # linear prediction coefficients
    a = lpc(x, len(lpc_mem))
    coeff = np.asarray(a[0])

    est_frames, lpc_mem = lfilter(0 - coeff, 1, x, -1, lpc_mem)
    res_frames = x - est_frames

    return coeff, lpc_mem, res_frames


def init_lpc_encoder(frame_length, nframes, original_data):
    original_data = original_data / max(original_data)
    original_data = np.multiply(original_data, 2 ** 15)
    original_data = np.ravel(original_data)

    lpcOrder = calculate_lpc_order(frame_length)

    amp = np.zeros(nframes)
    residual = np.zeros((nframes, frame_length))
    lpc_mem = np.zeros(lpcOrder)
    coeff = np.zeros((nframes, lpcOrder))
    idx = np.arange(frame_length)

    encode_data = np.zeros(np.dot(int(nframes), int(frame_length)))

    return original_data, lpcOrder, amp, residual, lpc_mem, coeff, idx, encode_data


def lpc_encoder(frame_length, nframes, original_data):
    original_data, lpcOrder, amp, residual, lpc_mem, coeff, idx, encode_data = init_lpc_encoder(frame_length, nframes,
                                                                                                original_data)

    for i in np.arange(0, nframes).reshape(-1):
        frames = original_data[idx]

        g = lpc_gain(frames)
        amp[i] = g

        a, lpc_mem, res_frames = lpc_filter(frames, lpc_mem)
        residual[i, :] = res_frames

        if (len(a) == len(coeff)):
            coeff[i, :] = a
        else:
            coeff[i, :] = a[0: len(a) - 1]

        encode_data[idx] = res_frames
        idx += frame_length

    encode_data = encode_data / 2 ** 15
    return lpcOrder, amp, residual, coeff, encode_data
