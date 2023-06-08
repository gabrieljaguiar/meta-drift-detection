import argparse
import itertools
import multiprocessing
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from river import naive_bayes
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingTreeClassifier
from tqdm import tqdm
from utils.queue import Queue
from utils import adaptiveADWIN

from train_generators import META_STREAM_SIZE, drifiting_streams, complex_drifts
from validation_generators import validation_drifting_streams
from utils import concept_drift
from utils.evaluator import Evaluator

parser = argparse.ArgumentParser(description="Meta-Drift target")


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


def find_nearest(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def getMetaData(chunk, driftPosition, delta_value):
    model = naive_bayes.GaussianNB()
    drift_detector = adaptiveADWIN.AdaptiveADWIN(delta=delta_value)  # default baseline

    range_for_drift = 500

    true_positive = 0
    false_positive = 0
    false_negative = 0
    distance_to_drift = 0

    warm_period = 100

    concept_drift_detected = False

    x, y = chunk

    last_closest_drift = 0

    for idx, (x, y) in enumerate(zip(x.queue, y.queue)):
        model.learn_one(x, y)
        y_predicted = model.predict_one(x)
        if idx > warm_period:
            drift_detector.update(1 if y == y_predicted else 0)

        if drift_detector.drift_detected:
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

    for i, (x, y) in tqdm(enumerate(g.take(META_STREAM_SIZE))):
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
            (
                distance_to_drift,
                true_positive,
                false_positive,
                false_negative,
            ) = getMetaData(
                (X_queue, y_queue),
                driftPosition,
                delta_value=0.2,
            )
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
            }
            meta_target_df.append(item)
            stride = 0
            chunck_idx += 1
        idx += 1

    print("Finished {}...".format(stream_name))

    # print(meta_target_df)

    return meta_target_df


META_STREAM_SIZE = 100000

out = Parallel(n_jobs=N_JOBS)(
    delayed(task)(i, 0.2) for i in enumerate(complex_drifts[70:])
)

meta_df = itertools.chain.from_iterable(out)

pd.DataFrame(meta_df).to_csv("{}".format(args.output), index=None)
