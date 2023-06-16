import argparse
import itertools
import multiprocessing
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from river import naive_bayes
from river.datasets import synth
from river.drift import adwin, binary, KSWIN
from river.tree import HoeffdingTreeClassifier
from tqdm import tqdm
from utils.queue import Queue
from utils import adaptiveADWIN

from train_generators import META_STREAM_SIZE, drifiting_streams, complex_drifts
from validation_generators import validation_drifting_streams
from utils import concept_drift
from utils.evaluator import Evaluator

parser = argparse.ArgumentParser(description="Meta-Drift target")

parser.add_argument(
    "--drift-detector", type=str, help="ADWIN, KSWIN, EDDM, HDDM", default="ADWIN"
)

parser.add_argument("--output", type=str, help="FIXED or META", default="results")

parser.add_argument("--mt", type=int, default=1500, help="Meta window size")

parser.add_argument(
    "--st", type=int, default=500, help="Stride of evaluation window size"
)


parser.add_argument("--n-jobs", type=int, default=-1, help="Number of multiple process")

args = parser.parse_args()


META_WINDOW_SIZE = args.mt
STRIDE_WINDOW = args.st
N_JOBS = multiprocessing.cpu_count() if args.n_jobs == -1 else args.n_jobs
DD_MODEL = args.drift_detector


def find_nearest(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def getMetaData(chunk, driftPosition, delta_value):
    model = naive_bayes.GaussianNB()
    # model = HoeffdingTreeClassifier()
    if DD_MODEL == "ADWIN":
        drift_detector = adaptiveADWIN.AdaptiveADWIN(
            delta=delta_value
        )  # default baseline
    if DD_MODEL == "EDDM":
        drift_detector = binary.EDDM(warm_start=0, alpha=delta_value, beta=delta_value)
    if DD_MODEL == "HDDM":
        drift_detector = binary.HDDM_A(
            drift_confidence=delta_value, warning_confidence=delta_value + 0.001
        )
    if DD_MODEL == "KSWIN":
        drift_detector = KSWIN(alpha=delta_value)

    range_for_drift = 500

    true_positive = 0
    false_positive = 0
    false_negative = 0
    distance_to_drift = 0

    warm_period = 100

    concept_drift_detected = False

    x, y = chunk

    last_closest_drift = 0

    evaluator = Evaluator(windowSize=500)

    for idx, (x, y) in enumerate(zip(x.queue, y.queue)):
        model.learn_one(x, y)
        y_predicted = model.predict_one(x)
        evaluator.addResult((x, y), model.predict_proba_one(x))
        if idx > warm_period:
            drift_detector.update(0 if y == y_predicted else 1)

        if drift_detector.drift_detected:
            # print(
            #    "Drift detected at {} and drift at {}. Distance = {}".format(
            #        idx, driftPosition, abs(idx - driftPosition)
            #    )
            # )
            closest_drift = driftPosition
            distance_to_drift += abs(idx - closest_drift)
            if (closest_drift > 0) and (
                (idx <= closest_drift + range_for_drift)
                and (idx >= closest_drift)
                and (last_closest_drift != closest_drift)
            ):
                concept_drift_detected = True
                last_closest_drift = closest_drift
                true_positive += 1

            else:
                false_positive += 1
        # if idx % 500 == 0:
        #    print("Accuracy {}".format(evaluator.getAccuracy()))
    if (driftPosition == 0) and (
        (true_positive + false_positive + false_negative == 0)
    ):
        true_positive += 1

    if (driftPosition != 0) and (not concept_drift_detected):
        false_negative += 1
    return distance_to_drift, true_positive, false_positive, false_negative


def task(arg, delta_value):
    global META_WINDOW_SIZE, STRIDE_WINDOW
    stream_id, g = arg
    range_for_drift = 500
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

    print("Running {}...".format(stream_name))

    X_queue = Queue(META_WINDOW_SIZE)
    y_queue = Queue(META_WINDOW_SIZE)

    number_of_drifts_detected = 0
    distance_to_drift = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    idx = 0

    stride = 0

    chunck_idx = 0

    meta_target_df = []

    for i, (x, y) in enumerate(g.take(META_STREAM_SIZE)):
        X_queue.insert(x)
        y_queue.insert(y)
        stride += 1

        if (
            X_queue.getNumberOfElements() == META_WINDOW_SIZE
            and stride >= STRIDE_WINDOW
        ):
            driftPosition = find_nearest(drift_positions, idx)
            if driftPosition > idx:
                driftPosition = 0
            else:
                driftPosition = max(driftPosition - (idx - META_WINDOW_SIZE) - 1, 0)

            if idx > 0:
                # print(
                #    "Getting meta_data for chunck {} <-> {}".format(
                #        idx - META_WINDOW_SIZE, idx
                #    )
                # )
                (
                    distance_to_drift,
                    true_positive,
                    false_positive,
                    false_negative,
                ) = getMetaData(
                    (X_queue, y_queue),
                    driftPosition,
                    delta_value=delta_value,
                )
                avg_dd = distance_to_drift / (
                    true_positive + false_positive + false_negative
                )
                detection_ratio = (false_negative + false_positive) / (
                    true_positive + 1
                )
                score = avg_dd * (10**detection_ratio)
                item = {
                    "stream": stream_name,
                    "chunk": chunck_idx,
                    "idx": idx,
                    "delta_value": delta_value,
                    "drift_position": driftPosition,
                    "detection_delay": distance_to_drift,
                    "tpr": true_positive,
                    "fnr": false_negative,
                    "fpr": false_positive,
                    "score": score,
                }
                meta_target_df.append(item)
                stride = 0
                chunck_idx += 1
        idx += 1

    print("Finished {}...".format(stream_name))

    # print(meta_target_df)

    return meta_target_df


META_STREAM_SIZE = 100000

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


meta_dataset = []

#

# kswin values = adwin values

# hddm_a = [0.01, 0.005, 0.002, 0.001, 0.0005]

# adwin values
if DD_MODEL == "ADWIN":
    possible_delta_values = [0.01, 0.008, 0.005, 0.002, 0.001]  # default baseline
if DD_MODEL == "EDDM":
    possible_delta_values = [0.99, 0.95, 0.9, 0.85, 0.8]
if DD_MODEL == "HDDM":
    possible_delta_values = [0.01, 0.005, 0.002, 0.001, 0.0005]
if DD_MODEL == "KSWIN":
    possible_delta_values = [0.01, 0.008, 0.005, 0.002, 0.001]  # default baseline


out = Parallel(n_jobs=N_JOBS)(
    delayed(task)(i, delta_value)
    for i, delta_value in list(
        itertools.product(enumerate(complex_drifts[:12]), possible_delta_values)
    )
)


meta_df = itertools.chain.from_iterable(out)
meta_dataset.append(meta_df)

meta_dataset = itertools.chain.from_iterable(meta_dataset)

df = pd.DataFrame(meta_dataset)

df.to_csv("{}".format(args.output), index=None)
