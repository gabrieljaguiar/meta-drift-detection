import argparse
import itertools
import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from river.drift import KSWIN, adwin, binary
from river.tree import HoeffdingAdaptiveTreeClassifier
from validation_generators import validation_drifting_streams

from utils import concept_drift
from utils.evaluator import Evaluator
from utils.queue import Queue


parser = argparse.ArgumentParser(description="Meta-Drift target")

parser.add_argument(
    "--drift-detector", type=str, help="ADWIN, KSWIN, EDDM, HDDM", default="ADWIN"
)

parser.add_argument("--output", type=str, help="FIXED or META", default="results")

parser.add_argument("--mt", type=int, default=1500, help="Meta window size")

parser.add_argument(
    "--st", type=int, default=500, help="Stride of evaluation window size"
)

parser.add_argument("--evaluation", default=False, action="store_true")


parser.add_argument("--n-jobs", type=int, default=-1, help="Number of multiple process")

args = parser.parse_args()


META_WINDOW_SIZE = args.mt
STRIDE_WINDOW = args.st
N_JOBS = multiprocessing.cpu_count() if args.n_jobs == -1 else args.n_jobs
DD_MODEL = args.drift_detector
EVALUATION = args.evaluation

IMBALANCE_SCENARIO = imbalance_ratios = {
    "0.5_0.5_0.5_0.5_0.5": "STABLE",
    "0.8_0.4_0.3_0.2_0.1": "INVERTED",
    "0.8_0.2_0.8_0.2_0.8": "FLIPPING",
    "0.5_0.25_0.1_0.25_0.5": "INCREASE_DECREASE",
}


def find_nearest(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def getMetaData(chunk, delta_value):
    if DD_MODEL == "ADWIN":
        drift_detector = adwin.ADWIN()
    if DD_MODEL == "EDDM":
        drift_detector = binary.EDDM()
    if DD_MODEL == "HDDM":
        drift_detector = binary.HDDM_W()
    if DD_MODEL == "KSWIN":
        drift_detector = KSWIN(seed=42)
    if DD_MODEL == "DDM":
        drift_detector = binary.DDM()

    model = HoeffdingAdaptiveTreeClassifier(drift_detector=drift_detector, seed=42)

    x_queue, y_queue = chunk

    for c in set(y_queue.queue):
        model.classes.add(c)

    evaluator = Evaluator(windowSize=500, numberOfClasses=len(model.classes))

    metrics = []
    idx = 0
    y = 0
    # print(x_queue.queue)
    for idx, (x, y) in enumerate(zip(x_queue.queue, y_queue.queue)):
        # print(idx)
        evaluator.addResult((x, y), model.predict_proba_one(x))
        model.learn_one(x, y)
        if (idx + 1) % 500 == 0:
            metrics.append(
                {
                    "idx": idx,
                    "acc": evaluator.getAccuracy(),
                    "gmean": evaluator.getGMean(),
                }
            )
        idx += 1

    return metrics


def task(arg, delta_value):
    global META_WINDOW_SIZE, META_STREAM_SIZE, STRIDE_WINDOW, IMBALANCE_SCENARIO
    stream_id, g = arg

    if isinstance(g, concept_drift.ConceptDriftStream):
        # print("here")
        drift_width = g.width
        # print(g.initialStream)
        stream_name = g.initialStream.generator.__class__.__name__

        drift_positions = []
        drift_position = 0
        next_stream = g
        size = g.size

        imb_scenario = ""
        imb_scenario = "{}".format(next_stream.initialStream.getImbalance())

        while isinstance(next_stream, concept_drift.ConceptDriftStream):
            drift_position += g.position

            drift_positions.append(drift_position)
            next_stream = next_stream.nextStream
            if isinstance(next_stream, concept_drift.ConceptDriftStream):
                imb_scenario = "{}_{}".format(
                    imb_scenario, next_stream.initialStream.getImbalance()
                )
            else:
                imb_scenario = "{}_{}".format(imb_scenario, next_stream.getImbalance())
        imb_scenario = IMBALANCE_SCENARIO.get(imb_scenario)
        stream_name = "{}_{}_{}_{}_{}".format(
            stream_id, stream_name, imb_scenario, size, drift_width
        )
        g.reset()
    else:
        size = g.size
        drift_positions = [size / 2]
        stream_name = g._repr_content.get("Name")

    idx = 0

    stride = 0

    chunk_idx = 0

    meta_target_df = []

    size = size

    print(
        "Evaluating {} with {} for delta {}".format(stream_name, DD_MODEL, delta_value)
    )

    META_WINDOW_SIZE = size

    X_queue = Queue(META_WINDOW_SIZE)
    y_queue = Queue(META_WINDOW_SIZE)

    for i, (x, y) in enumerate(g.take(size)):
        X_queue.insert(x)
        y_queue.insert(y)
        stride += 1

        if (
            X_queue.getNumberOfElements() == META_WINDOW_SIZE
            and stride >= STRIDE_WINDOW
        ):
            # print("entrou if")
            driftPosition = find_nearest(drift_positions, idx)
            if driftPosition > idx:
                driftPosition = 0
            else:
                driftPosition = max(driftPosition - (idx - META_WINDOW_SIZE) - 1, 0)

            if driftPosition != 0:
                metrics = getMetaData((X_queue, y_queue), delta_value)

                metrics_df = pd.DataFrame(metrics)

                item = {
                    "stream": stream_name,
                    "chunk": chunk_idx,
                    "idx": idx,
                    "size": size,
                    "drift_position": driftPosition,
                    "delta_value": delta_value,
                    "acc": metrics_df["acc"].mean(),
                    "gmean": metrics_df["gmean"].mean(),
                }

                metrics_df.to_csv("./metrics/{}_{}.csv".format(DD_MODEL, stream_name))

                meta_target_df.append(item)

            stride = 0
            chunk_idx += 1
        idx += 1

    print(
        "Finished evaluating {} with {} for delta {}".format(
            stream_name, DD_MODEL, delta_value
        )
    )

    return meta_target_df


meta_dataset = []
possible_delta_values = [1]


out = Parallel(n_jobs=N_JOBS)(
    delayed(task)(i, delta_value)
    for i, delta_value in list(
        itertools.product(enumerate(validation_drifting_streams), possible_delta_values)
    )
)

meta_df = itertools.chain.from_iterable(out)
meta_dataset.append(meta_df)

meta_dataset = itertools.chain.from_iterable(meta_dataset)

df = pd.DataFrame(meta_dataset)

df.to_csv("{}".format(args.output), index=None)
