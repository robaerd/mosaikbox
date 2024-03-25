import numpy as np


def plomp(f1, f2):
    """
      Plomp's algorithm for estimating roughness.

    :param f1:  float, frequency of first frequency of the pair
    :param f2:  float, frequency of second frequency of the pair
    :return:
    """
    b1 = 3.51
    b2 = 5.75
    xstar = 0.24
    s1 = 0.0207
    s2 = 18.96
    s = np.tril(xstar / ((s1 * np.minimum(f1, f2)) + s2))
    pd = np.exp(-b1 * s * np.abs(f2 - f1)) - np.exp(-b2 * s * np.abs(f2 - f1))
    return pd


def detect_peaks(array, freq=0, cthr=0.2, unprocessed_array=False, fs=44100):
    """
      Function detects the peaks in array, based from the mirpeaks algorithm.

    :param array:               Array in which to detect peaks
    :param freq:                Scale representing the x axis (sample length as array)
    :param cthr:                Threshold for checking adjacent peaks
    :param unprocessed_array:   Array that in unprocessed (normalised), if False will default to the same as array.
    :param fs:                  Sampe rate of the array

    :return:                     index of peaks, values of peaks, peak value on freq.

    Copyright 2018 Andy Pearce, Institute of Sound Recording, University of Surrey, UK.

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
    """
    # flatten the array for correct processing
    array = array.flatten()

    if np.isscalar(freq):
        # calculate the frerquency scale - assuming a samplerate if none provided
        freq = np.linspace(0, fs / 2.0, len(array))

    if np.isscalar(unprocessed_array):
        unprocessed_array = array

    # add values to allow peaks at the first and last values
    array_appended = np.insert(array, [0, len(array)], -2.0)  # to allow peaks at start and end (default of mir)
    # unprocessed array to get peak values
    array_unprocess_appended = np.insert(unprocessed_array, [0, len(unprocessed_array)], -2.0)
    # append the frequency scale for precise freq calculation
    freq_appended = np.insert(freq, [0, len(freq)], -1.0)

    # get the difference values
    diff_array = np.diff(array_appended)

    # find local maxima
    mx = np.array(np.where((array >= cthr) & (diff_array[0:-1] > 0) & (diff_array[1:] <= 0))) + 1

    # initialise arrays for output
    finalmx = []
    peak_value = []
    peak_x = []
    peak_idx = []

    if np.size(mx) > 0:
        # unpack the array if peaks found
        mx = mx[0]

        j = 0  # scans the peaks from beginning to end
        mxj = mx[j]  # the current peak under evaluation
        jj = j + 1
        bufmin = 2.0
        bufmax = array_appended[mxj]

        if mxj > 1:
            oldbufmin = min(array_appended[:mxj - 1])
        else:
            oldbufmin = array_appended[0]

        while jj < len(mx):
            # if adjacent mx values are too close, returns no array
            if mx[jj - 1] + 1 == mx[jj] - 1:
                bufmin = min([bufmin, array_appended[mx[jj - 1]]])
            else:
                bufmin = min([bufmin, min(array_appended[mx[jj - 1]:mx[jj] - 1])])

            if bufmax - bufmin < cthr:
                # There is no contrastive notch
                if array_appended[mx[jj]] > bufmax:
                    # new peak is significant;y higher than the old peak,
                    # the peak is transfered to the new position
                    j = jj
                    mxj = mx[j]  # the current peak
                    bufmax = array_appended[mxj]
                    oldbufmin = min([oldbufmin, bufmin])
                    bufmin = 2.0
                elif array_appended[mx[jj]] - bufmax <= 0:
                    bufmax = max([bufmax, array_appended[mx[jj]]])
                    oldbufmin = min([oldbufmin, bufmin])

            else:
                # There is a contrastive notch
                if bufmax - oldbufmin < cthr:
                    # But the previous peak candidate is too weak and therefore discarded
                    oldbufmin = min([oldbufmin, bufmin])
                else:
                    # The previous peak candidate is OK and therefore stored
                    finalmx.append(mxj)
                    oldbufmin = bufmin

                bufmax = array_appended[mx[jj]]
                j = jj
                mxj = mx[j]  # The current peak
                bufmin = 2.0

            jj += 1
        if bufmax - oldbufmin >= cthr and (bufmax - min(array_appended[mx[j] + 1:]) >= cthr):
            # The last peak candidate is OK and stored
            finalmx.append(mx[j])

        ''' Sort the values according to their level '''
        finalmx = np.array(finalmx)
        sort_idx = np.argsort(array_appended[finalmx])[::-1]  # descending sort
        finalmx = finalmx[sort_idx]

        peak_idx = finalmx - 1  # indexes were for the appended array, -1 to return to original array index
        peak_value = array_unprocess_appended[finalmx]
        peak_x = freq_appended[finalmx]

        ''' Interpolation for more precise peak location '''
        corrected_value = []
        corrected_position = []
        for current_peak_idx in finalmx:
            # if there enough space to do the fitting
            if 1 < current_peak_idx < (len(array_unprocess_appended) - 2):
                y0 = array_unprocess_appended[current_peak_idx]
                ym = array_unprocess_appended[current_peak_idx - 1]
                yp = array_unprocess_appended[current_peak_idx + 1]
                p = (yp - ym) / (2 * (2 * y0 - yp - ym))
                corrected_value.append(y0 - (0.25 * (ym - yp) * p))
                if p >= 0:
                    correct_pos = ((1 - p) * freq_appended[current_peak_idx]) + (
                                p * freq_appended[current_peak_idx + 1])
                    corrected_position.append(correct_pos)
                elif p < 0:
                    correct_pos = ((1 + p) * freq_appended[current_peak_idx]) - (
                                p * freq_appended[current_peak_idx - 1])
                    corrected_position.append(correct_pos)
            else:
                corrected_value.append(array_unprocess_appended[current_peak_idx])
                corrected_position.append(freq_appended[current_peak_idx])

        if corrected_position:
            peak_x = corrected_position
            peak_value = corrected_value

    return peak_idx, peak_value, peak_x


def output_clip(score, min_score=0, max_score=100):
    """
      Limits the output of the score between min_score and max_score

    :param score:
    :param min_score:
    :param max_score:
    :return:
    """
    if score < min_score:
        return 0.0
    elif score > max_score:
        return 100.0
    else:
        return score


def timbral_roughness(audio_samples, fs, bands: list[tuple[float, float]], dev_output=False, phase_correction=False,
                      clip_output=False, peak_picking_threshold=0.01):
    """
     Modified by Robert Sowula to support bands

     This function is an implementation of the Vassilakis [2007] model of roughness.
     The peak picking algorithm implemented is based on the MIR toolbox's implementation.

     This version of timbral_roughness contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4


     Vassilakis, P. 'SRA: A Aeb-based researh tool for spectral and roughness analysis of sound signals', Proceedings
     of the 4th Sound and Music Computing Conference, Lefkada, Greece, July, 2007.

     Required parameter
      :param audio_samples:     audio_samples, audio numpy array with values between [-1, 1]
      :param fs:                fs, frame rate (frequency of sampling) of the audio
      :param bands:             the frequency bands to perform roughness calculation on

     Optional parameters
      :param dev_output:            bool, when False return the roughness, when True return all extracted features
                                    (current none).
      :param phase_correction:      bool, if the inter-channel phase should be estimated when performing a mono sum.
                                    Defaults to False.

      :return:                      Roughness of the audio signal.

     Copyright 2018 Andy Pearce, Institute of Sound Recording, University of Surrey, UK.

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
    """

    '''
      Pad audio
    '''
    # pad audio
    audio_samples = np.lib.pad(audio_samples, (512, 0), 'constant', constant_values=(0.0, 0.0))

    '''
      Reshape audio into time windows of 50ms.
    '''
    # reshape audio
    audio_len = len(audio_samples)
    time_step = 0.05
    step_samples = int(fs * time_step)
    nfft = step_samples
    window = np.hamming(nfft + 2)
    window = window[1:-1]
    olap = nfft / 2
    num_frames = int((audio_len) / (step_samples - olap))
    next_pow_2 = np.log(step_samples) / np.log(2)
    next_pow_2 = 2 ** int(next_pow_2 + 1)

    reshaped_audio = np.zeros([next_pow_2, num_frames])

    i = 0
    start_idx = int((i * (nfft / 2.0)))

    # check if audio is too short to be reshaped
    if audio_len > step_samples:
        # get all the audio
        while start_idx + step_samples <= audio_len:
            audio_frame = audio_samples[start_idx:start_idx + step_samples]

            # apply window
            audio_frame = audio_frame * window

            # append zeros
            reshaped_audio[:step_samples, i] = audio_frame

            # increase the step
            i += 1
            start_idx = int((i * (nfft / 2.0)))
    else:
        # reshaped audio is just padded audio samples
        reshaped_audio[:audio_len, i] = audio_samples

    spec = np.fft.fft(reshaped_audio, axis=0)
    spec_len = int(next_pow_2 / 2) + 1
    spec = spec[:spec_len, :]
    spec = np.absolute(spec)

    freq = fs / 2 * np.linspace(0, 1, spec_len)

    # normalise spectrogram based from peak TF bin
    norm_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

    ''' Peak picking algorithm '''
    cthr = peak_picking_threshold  # threshold for peak picking

    _, no_segments = np.shape(spec)

    allpeakpos = []
    allpeaklevel = []
    allpeaktime = []

    for i in range(0, no_segments):
        d = norm_spec[:, i]
        d_un = spec[:, i]

        # find peak candidates
        peak_pos, peak_level, peak_x = detect_peaks(d, cthr=cthr, unprocessed_array=d_un, freq=freq, fs=fs)

        allpeakpos.append(peak_pos)
        allpeaklevel.append(peak_level)
        allpeaktime.append(peak_x)

    ''' Calculate the Vasillakis Roughness for each band '''
    band_roughness = []
    for band_start, band_end in bands:
        allroughness = []
        for frame in range(len(allpeaklevel)):
            frame_freq = np.array(allpeaktime[frame])
            frame_level = np.array(allpeaklevel[frame])

            mask = (frame_freq >= band_start) & (frame_freq <= band_end)
            band_freq = frame_freq[mask]
            band_level = frame_level[mask]

            if len(band_freq) > 1:
                f2 = np.kron(np.ones([len(band_freq), 1]), band_freq)
                f1 = f2.T
                v2 = np.kron(np.ones([len(band_level), 1]), band_level)
                v1 = v2.T

                X = v1 * v2
                Y = (2 * v2) / (v1 + v2)
                Z = plomp(f1, f2)
                rough = (X ** 0.1) * (0.5 * (Y ** 3.11)) * Z

                allroughness.append(np.sum(rough))
            else:
                allroughness.append(0)

        mean_roughness = np.mean(allroughness)
        band_roughness.append(mean_roughness)

    ''' Perform the specified post-processing on the average roughness for each band '''
    processed_roughness = []
    for mean_roughness in band_roughness:
        if not dev_output:
            # cap roughness for low end
            if mean_roughness < 0.01:
                processed_roughness.append(0)
            else:
                roughness = np.log10(mean_roughness) * 13.98779569 + 48.97606571545886
                if clip_output:
                    roughness = output_clip(roughness)

                processed_roughness.append(roughness)
        else:
            processed_roughness.append(mean_roughness)

    return np.array(processed_roughness)
