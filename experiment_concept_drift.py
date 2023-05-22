from tqdm import tqdm
from utils import concept_drift
from utils import adaptiveADWIN
from utils.queue import Queue
from river.tree import HoeffdingTreeClassifier
from meta_features import extract_meta_features
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from validation_generators import validation_drifting_streams, META_STREAM_SIZE

import os
import pandas as pd
import numpy as np


def find_nearest(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


MODEL = "META"  # FIXED for fixed adwin value or META for meta stream
META_WINDOW_SIZE = 1500
STRIDE_WINDOW = 500


meta_target_df = pd.read_csv("meta_target.csv")

meta_target_filtered = meta_target_df.loc[
    meta_target_df.groupby("stream").score.idxmin()
].reset_index(drop=True)

filling_imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0
)

if MODEL == "META":
    training_meta_features = pd.read_csv("./training_meta_features.csv")

    training_meta_features = training_meta_features.fillna(0)

    meta_target = meta_target_filtered.loc[:, ["stream", "delta_value"]]

    meta_dataset = training_meta_features.merge(
        right=meta_target, how="left", left_on="stream_name", right_on="stream"
    )

    meta_dataset.drop("stream_name", axis=1, inplace=True)

    idx_column = "stream"
    class_column = "delta_value"

    meta_model = RandomForestRegressor(random_state=42)

    meta_model.fit(
        X=meta_dataset.loc[
            :, meta_dataset.columns.difference([idx_column, class_column])
        ],
        y=meta_dataset.loc[:, class_column],
    )

    if not os.path.exists("meta_dataset.csv"):
        meta_dataset.to_csv("meta_dataset.csv", index=None)

    mfe_feature_list = [
        "joint_ent",
        "ns_ratio",
        "can_cor",
        "gravity",
        "kurtosis",
        "skewness",
        "sparsity",
        "f1",
        "f1v",
        "f2",
        "f3",
        "f4",
        "n1",
        "n2",
    ]

    summary = ["mean", "sd"]
    tsfel_config = {}

range_for_drift = 100

results = []


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
        X_queue = Queue(META_WINDOW_SIZE)
        y_queue = Queue(META_WINDOW_SIZE)

    model = HoeffdingTreeClassifier()

    drift_detector = adaptiveADWIN.AdaptiveADWIN(delta=0.5)  # default baseline

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

    stride = 0

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
            X_queue.insert(x)
            y_queue.insert(y)
            stride += 1

            if (
                X_queue.getNumberOfElements() == META_WINDOW_SIZE
                and stride >= STRIDE_WINDOW
            ):
                stride = 0
                meta_features = extract_meta_features(
                    pd.DataFrame(X_queue.getQueue()),
                    pd.DataFrame(y_queue.getQueue()),
                    summary=summary,
                    tsfel_config=tsfel_config,
                    mfe_feature_config=mfe_feature_list,
                )
                meta_features_df = pd.DataFrame(meta_features)
                meta_features_df.fillna(0, inplace=True)
                adwin_prediction = meta_model.predict(meta_features_df)
                drift_detector.updateDelta(adwin_prediction)

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
        "model": MODEL,
        "drift_position": drift_position,
        "detection_delay": distance_to_drift,
        "tpr": true_positive,
        "fnr": false_negative,
        "fpr": false_positive,
    }

    results.append(item)


pd.DataFrame(results).to_csv("results_{}.csv".format(MODEL))
