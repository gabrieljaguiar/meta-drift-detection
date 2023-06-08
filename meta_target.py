import itertools
import multiprocessing
import os

import pandas as pd
from joblib import Parallel, delayed
from river import naive_bayes
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingTreeClassifier
from tqdm import tqdm

from train_generators import META_STREAM_SIZE, drifiting_streams
from utils import concept_drift
from utils.evaluator import Evaluator

# formula avgdd = dd/number_of_detected_drifts #lower the better
# detection_ratio = (fnr + fpr) / (tpr + 1) # lower the better
# avgdd*10^detection_ratio #lower the better


EVALUATION_WINDOW = 500

N_JOBS = 2 if multiprocessing.cpu_count() <= 2 else min(multiprocessing.cpu_count(), 48)


def task(arg, delta_value):
    global range_for_drift
    stream_id, g = arg

    if isinstance(g, concept_drift.ConceptDriftStream):
        drift_position = g.position
        drift_width = g.width
        stream_identifier = g.initialStream._repr_content.get("Name")

        sizes = [g.size]
        range_for_drift = max(range_for_drift, drift_width)
        g.reset()
    else:
        drift_position = 0
        drift_width = 1
        stream_identifier = g._repr_content.get("Name")

        sizes = META_STREAM_SIZE

    meta_samples = []

    for size in sizes:
        model = naive_bayes.GaussianNB()
        drift_detector = adwin.ADWIN(delta=delta_value)

        grace_period = int(size * 0.05)
        stream_name = "{}_{}_{}_{}_{}".format(
            stream_id, stream_identifier, drift_position, drift_width, size
        )

        print("Running {}...".format(stream_name))

        number_of_drifts_detected = 0
        distance_to_drift = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0
        idx = 0

        for x, y in g.take(size):
            y_hat = model.predict_proba_one(x)
            y_predicted = model.predict_one(x)

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
            distance_to_drift = size

        if (drift_position == 0) and (
            (true_positive + false_positive + false_negative == 0)
        ):  # There is no drift and no drift was detected
            true_positive += 1

        avg_dd = distance_to_drift / (true_positive + false_positive + false_negative)
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

        print("Finished {}...".format(stream_name))

        meta_samples.append(item)

    return meta_samples


if not os.path.exists("meta_target.csv"):
    window_size = 500
    idx = 0
    meta_dataset = []

    possible_delta_values = [
        (1 / i)
        for i in [
            2,
            5,
            8,
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

    meta_dataset = []

    for delta_value in possible_delta_values:
        print("Evaluating for delta = {}".format(delta_value))

        meta_df = Parallel(n_jobs=N_JOBS)(
            delayed(task)(i, delta_value) for i in enumerate(drifiting_streams)
        )

        meta_df = itertools.chain.from_iterable(meta_df)

        meta_dataset.append(meta_df)

    meta_dataset = itertools.chain.from_iterable(meta_dataset)

    df = pd.DataFrame(meta_dataset)
    df.to_csv("meta_target.csv", index=False)
else:
    meta_target_df = pd.read_csv("meta_target.csv")

    meta_target_filtered = meta_target_df.loc[
        meta_target_df.groupby("stream").score.idxmin()
    ].reset_index(drop=True)
