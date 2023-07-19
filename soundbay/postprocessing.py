import numpy as np
import pandas as pd
import librosa


def pp_accurate_segment_edges_factory(postprocessor_name, **kwargs) -> dict:
    '''
    This factory wraps the execution of all postprocessing options we have, aims to find more precise start and end time.
    All the functions here return the same dict with the following info:
    {'updated_start_time': X[SEC],
    'confidence_start': X_c \in [0,1],
    'updated_end_time': Y,
    'confidence_end': Y_c}
    '''
    est_start, conf_start, est_end, conf_end = {
        'energy': energy_based_det,
        'covariance': covariance_based_det,
        'rms': rms,
        'magnitude': magnitude_time_domain,
    }[postprocessor_name](**kwargs)

    return {'estimated_start_time': est_start,
            'confidence_start': conf_start,
            'estimated_end_time': est_end,
            'confidence_end': conf_end}


def postprocessor(audio_path, results_df, seq_len, TH_positive:float=0.5, postprocessors: list=[]) -> pd.DataFrame:
    length_signal, num_classes = results_df.shape
    # length_signal_seconds = seq_len*length_signal  # [segments]*[sec/segment]

    if num_classes > 2:
        print(
            f'postprocessing function currently supports only 2 classes (detection),'
            f'but this signal has {num_classes} classes. It returns the df as is')
        return results_df
    # else: == we have only 2 classes: {bg,call}
    list_of_potential_seqs = find_potential_segments(results_df, TH_positive)  # units - indices
    list_of_potential_seqs_secs = [[start*seq_len, end*seq_len] for sub_l in list_of_potential_seqs for (start, end) in sub_l]

    new_df, pp_signals = pd.DataFrame, dict()
    for start_time, end_time in list_of_potential_seqs_secs:
        # Load the specified segment of the audio signal
        samples, sample_rate = librosa.load(audio_path, sr=None, offset=start_time, duration=end_time-start_time)
        # Apply with the desired postprocessing algorithms
        for key in postprocessors:
            pp_signals[key] = pp_accurate_segment_edges_factory(key, sample=samples, sr=sample_rate, )
        new_df.append({'raw_start_time': start_time, 'raw_end_time': end_time, **pp_signals})


def find_potential_segments(signal: pd.DataFrame, TH_positive: float=0.5) -> list:
    # find adjacent segments that are potentially related to the same sequence
    list_of_potential_seqs = []
    potential_segment = None
    for i in range(len(signal)):
        sample = signal.iloc[i]
        if sample[1] > sample[0] and sample[1] > TH_positive:  # this chunk is to be considered as calling
            curr_start, curr_end = i, i + 1
            if potential_segment is None:
                potential_segment = [curr_start - 1, curr_end]  # start time is going one back to cover the accurate start point for sure
            else:
                potential_segment[1] = curr_end  # increasing the end time only
        else:  # the current sample is not considered as call
            if potential_segment:
                list_of_potential_seqs.append(potential_segment)
                potential_segment = None
            else:  # nothing has been considered as "call" yet
                pass
    # taking care of the case when the dataframe ends with a call and then return the final list
    return list_of_potential_seqs.append(potential_segment) if potential_segment else list_of_potential_seqs


def energy_based_det(*, potential_seq: np.array) -> (float, float):
    pass
    # return [estimated_start_time, confidence_start, estimated_end_time, confidence_end]


def covariance_based_det(*, potential_seq: np.array) -> dict:
    pass


def rms(*, potential_seq: np.array) -> dict:
    pass


def magnitude_time_domain(*, potential_seq: np.array) -> dict:
    pass