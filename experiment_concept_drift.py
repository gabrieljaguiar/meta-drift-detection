# HERE IMPORT GENERATED VALIDATION DATA STREAMS
# TRAIN A MODEL ON META_FEATURES
# EVAL THEM USING HT AND ADWIN VALUES GIVEN BY THE MODEL
# EVAL THE PERFORMANCE OF CONCEPT DRIFT DETECTOR WITH MAJORITY VALUE (BASELINE)
from utils import concept_drift
import numpy as np
from river.drift import adwin, binary
from river.tree import HoeffdingTreeClassifier
from validation_generators import validation_drifting_streams, META_STREAM_SIZE
import pandas as pd
from tqdm import tqdm
import os
from utils import adaptiveADWIN


def find_nearest(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


MODEL = "FIXED"  # FIXED for fixed adwin value or META for meta stream


meta_target_df = pd.read_csv("meta_target.csv")

meta_target_filtered = meta_target_df.loc[
    meta_target_df.groupby("stream").score.idxmin()
].reset_index(drop=True)

range_for_drift = 100


for stream_id, g in tqdm(
    enumerate(validation_drifting_streams), total=len(validation_drifting_streams)
):
    if isinstance(g, concept_drift.ConceptDriftStream):
        drift_width = g.width
        stream_name = g.initialStream._repr_content.get("Name")
        stream_name = "{}_{}_20000_{}".format(stream_id, stream_name, drift_width)
        range_for_drift = max(range_for_drift, drift_width)
        drift_positions = []
        drift_position = 0
        next_stream = g
        while isinstance(next_stream, concept_drift.ConceptDriftStream):
            drift_position += g.position
            drift_positions.append(drift_position)
            next_stream = next_stream.nextStream

        print(drift_positions)
        g.reset()
    else:
        drift_positions = []
        drift_position = 0
        drift_width = 1
        range_for_drift = 0
        stream_name = g._repr_content.get("Name")
        stream_name = "{}_{}_{}_{}".format(
            stream_id, stream_name, drift_position, drift_width
        )

    if MODEL == "META":
        evaluation_window = []

    model = HoeffdingTreeClassifier()

    drift_detector = adaptiveADWIN.AdaptiveADWIN(delta=0.1)

    number_of_drifts_detected = 0
    distance_to_drift = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    grace_period = int(META_STREAM_SIZE * 0.05)

    idx = 0

    previous_true_positive = 0

    concept_drift_detected = False

    next_drift_idx = 0

    try:
        next_drift = drift_positions[next_drift_idx]
    except IndexError:
        next_drift = 0

    for x, y in g.take(META_STREAM_SIZE):
        if (idx > (next_drift + range_for_drift)) and (next_drift > 0):
            next_drift_idx += 1
            try:
                next_drift = drift_positions[next_drift_idx]
            except IndexError:
                next_drift = META_STREAM_SIZE

            if not concept_drift_detected:
                false_negative += 1

            correct_drift_detected = False

        if MODEL == "META":
            # Append to the list, control size of list, stride to extract meta_features and update ADWIN delta with the meta-model
            evaluation_window.append(x)

        y_hat = model.predict_proba_one(x)
        y_predicted = model.predict_one(x)

        model.learn_one(x, y)
        if idx >= grace_period:
            drift_detector.update(1 if y == y_predicted else 0)

        if drift_detector.drift_detected:
            closest_drift = find_nearest(drift_positions, idx)
            distance_to_drift += abs(idx - closest_drift)
            number_of_drifts_detected += 1
            if (closest_drift > 0) and (
                (idx <= closest_drift + range_for_drift)
                and (idx >= closest_drift - range_for_drift)
            ):
                concept_drift_detected = True
                true_positive += 1
            else:
                false_positive += 1

        idx += 1

    item = {
        "stream": stream_name,
        "drift_position": drift_position,
        "detection_delay": distance_to_drift,
        "tpr": true_positive,
        "fnr": false_negative,
        "fpr": false_positive,
    }
