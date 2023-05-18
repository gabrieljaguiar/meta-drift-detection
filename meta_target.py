from utils import concept_drift
from utils import evaluator
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingTreeClassifier
from train_generators import drifiting_streams, META_STREAM_SIZE
import pandas as pd
from tqdm import tqdm
import os

# formula avgdd = dd/number_of_detected_drifts #lower the better
# detection_ratio = (fnr + fpr) / (tpr + 1) # lower the better
# avgdd*10^detection_ratio #lower the better

if not os.path.exists("meta_target.csv"):
    window_size = 500
    idx = 0
    meta_dataset = []
    grace_period = int(META_STREAM_SIZE * 0.05)
    possible_delta_values = [
        (1 / i)
        for i in [
            10,
            20,
            30,
            40,
            50,
            100,
            200,
            300,
            400,
            500,
            1000,
            5000,
            10000,
            50000,
            100000,
            500000,
            10000000,
        ]
    ]

    range_for_drift = 100

    for delta_value in possible_delta_values:
        print("Evaluating for delta = {}".format(delta_value))
        for stream_id, g in tqdm(
            enumerate(drifiting_streams), total=len(drifiting_streams)
        ):
            streamEvaluator = evaluator.Evaluator(windowSize=window_size)
            model = HoeffdingTreeClassifier()
            drift_detector = adwin.ADWIN(delta=delta_value)
            idx = 0

            if isinstance(g, concept_drift.ConceptDriftStream):
                drift_position = g.position
                drift_width = g.width
                stream_name = g.initialStream._repr_content.get("Name")
                stream_name = "{}_{}_{}_{}".format(
                    stream_id, stream_name, drift_position, drift_width
                )
                range_for_drift = max(range_for_drift, drift_width)
                g.reset()
            else:
                drift_position = 0
                drift_width = 1
                stream_name = g._repr_content.get("Name")
                stream_name = "{}_{}_{}_{}".format(
                    stream_id, stream_name, drift_position, drift_width
                )

            number_of_drifts_detected = 0
            distance_to_drift = 0

            true_positive = 0
            false_positive = 0
            false_negative = 0

            stream_data = []

            for x, y in g.take(META_STREAM_SIZE):
                y_hat = model.predict_proba_one(x)
                y_predicted = model.predict_one(x)
                stream_data.append(y)

                streamEvaluator.addResult((x, y), y_hat)
                model.learn_one(x, y)
                if idx >= grace_period:
                    drift_detector.update(1 if y == y_predicted else 0)

                if drift_detector.drift_detected:
                    distance_to_drift += abs(idx - drift_position)
                    number_of_drifts_detected += 1
                    if (
                        (drift_position > 0)
                        and (
                            (idx <= drift_position + range_for_drift)
                            and (idx >= drift_position - range_for_drift)
                        )
                        and true_positive == 0
                    ):
                        true_positive += 1
                    else:
                        false_positive += 1

                idx += 1

            if (drift_position > 0) and (true_positive == 0) and (false_positive == 0):
                false_negative += 1
                distance_to_drift = META_STREAM_SIZE

            if (drift_position == 0) and (
                (true_positive + false_positive + false_negative == 0)
            ):  # There is no drift and no drift was detected
                true_positive += 1

            avg_dd = distance_to_drift / (
                true_positive + false_positive + false_negative
            )
            detection_ratio = (false_negative + false_positive) / (true_positive + 1)
            score = avg_dd * (10**detection_ratio)

            item = {
                "stream": stream_name,
                "drift_position": drift_position,
                "detection_delay": distance_to_drift,
                "tpr": true_positive,
                "fnr": false_negative,
                "fpr": false_positive,
                "delta_value": delta_value,
                "score": score,
            }

            meta_dataset.append(item)

        df = pd.DataFrame(meta_dataset)
        df.to_csv("meta_target.csv", index=False)
else:
    meta_target_df = pd.read_csv("meta_target.csv")

    meta_target_filtered = meta_target_df.loc[
        meta_target_df.groupby("stream").score.idxmin()
    ].reset_index(drop=True)

    print(meta_target_filtered["delta_value"].value_counts())
