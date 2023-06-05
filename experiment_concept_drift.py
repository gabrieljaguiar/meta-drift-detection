from joblib import Parallel, delayed
from utils import concept_drift
from utils import adaptiveADWIN
from utils.queue import Queue
from utils.evaluator import Evaluator
from river import naive_bayes
from meta_features import extract_meta_features
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from validation_generators import validation_drifting_streams, META_STREAM_SIZE

import os
import pandas as pd
import numpy as np
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description="Meta-Drift evaluation")


parser.add_argument("--model", type=str, help="FIXED or META", default="FIXED")

parser.add_argument(
    "--mf",
    type=str,
    help="Meta-feature file to be used",
    default="training_meta_features_set_1.csv",
)

parser.add_argument("--output", type=str, help="FIXED or META", default="results")

parser.add_argument("--ew", type=int, default=500, help="Evaluation window size")

parser.add_argument(
    "--st", type=int, default=500, help="Stride of evaluation window size"
)

parser.add_argument("--mt", type=int, default=1500, help="Meta window size")

parser.add_argument(
    "--save-metrics",
    default=False,
    help="If metrics file should be created",
    action="store_true",
)

parser.add_argument("--n-jobs", type=int, default=-1, help="Number of multiple process")

args = parser.parse_args()


MODEL = args.model  # FIXED for fixed adwin value or META for meta stream
META_WINDOW_SIZE = args.mt
STRIDE_WINDOW = args.st
EVALUATION_WINDOW = args.ew
N_JOBS = multiprocessing.cpu_count() if args.n_jobs == -1 else args.n_jobs

print("Starting experiment")
print("MODEL: {}".format(MODEL))
print("OUTPUT FILE: {}".format(args.output))
print("META-WINDOW: {}".format(META_WINDOW_SIZE))
print("STRIDE_WINDOW: {}".format(STRIDE_WINDOW))
print("EVALUATION_WINDOW: {}".format(EVALUATION_WINDOW))
print("PARALLALEL JOBS: {}".format(N_JOBS))


def find_nearest(array, value):
    if len(array) == 0:
        return 0
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def task(arg):
    global MODEL, META_WINDOW_SIZE, STRIDE_WINDOW, EVALUATION_WINDOW
    stream_id, g = arg
    range_for_drift = 100
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

    print("Running {}...".format(stream_name))

    if MODEL == "META":
        X_queue = Queue(META_WINDOW_SIZE)
        y_queue = Queue(META_WINDOW_SIZE)

    # model = HoeffdingTreeClassifier()

    # model = KNNClassifier(n_neighbors=5)

    model = naive_bayes.GaussianNB()
    drift_detector = adaptiveADWIN.AdaptiveADWIN(delta=0.2)  # default baseline

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

    stream_results = []

    evaluator = Evaluator(
        EVALUATION_WINDOW, g.n_classes if g.n_classes is not None else 2
    )

    last_closest_drift = 0

    for i, (x, y) in enumerate(g.take(META_STREAM_SIZE)):
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
                    summary_mfe=summary_mfe,
                    summary_tsfel=summary_tsfel,
                    tsfel_config=tsfel_config,
                    mfe_feature_config=mfe_feature_list,
                )

                meta_features_df = pd.DataFrame(meta_features)
                meta_features_df.fillna(0, inplace=True)
                adwin_prediction = meta_model.predict(
                    meta_features_df.loc[:, feature_columns]
                )
                drift_detector.updateDelta(adwin_prediction)

        y_hat = model.predict_proba_one(x)
        y_predicted = model.predict_one(x)
        evaluator.addResult((x, y), y_hat)

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
                and (last_closest_drift != closest_drift)
            ):
                concept_drift_detected = True
                last_closest_drift = closest_drift
                true_positive += 1

            else:
                false_positive += 1

        if (idx + 1) % EVALUATION_WINDOW == 0:
            eval_item = {"idx": idx, "accuracy": evaluator.getAccuracy()}
            stream_results.append(eval_item)

        idx += 1
    if args.save_metrics:
        metrics_df = pd.DataFrame(stream_results)
        metrics_df.to_csv("./metrics/{}.csv".format(stream_name))

    item = {
        "stream": stream_name,
        "model": MODEL,
        "drift_position": drift_position,
        "detection_delay": distance_to_drift,
        "tpr": true_positive,
        "fnr": false_negative,
        "fpr": false_positive,
    }

    print("Finished {}...".format(stream_name))

    return item


meta_target_df = pd.read_csv("meta_target.csv")

meta_target_filtered = meta_target_df.loc[
    meta_target_df.groupby("stream").score.idxmin()
].reset_index(drop=True)

filling_imputer = SimpleImputer(
    missing_values=np.nan, strategy="constant", fill_value=0
)

if MODEL == "META":
    print("META_FEATURE FILE: {}".format(args.mf))
    training_meta_features = pd.read_csv("./{}".format(args.mf))

    training_meta_features = training_meta_features.fillna(0)

    meta_target = meta_target_filtered.loc[:, ["stream", "delta_value"]]

    meta_dataset = training_meta_features.merge(
        right=meta_target, how="left", left_on="stream_name", right_on="stream"
    )

    meta_dataset.drop("stream_name", axis=1, inplace=True)

    idx_column = "stream"
    class_column = "delta_value"

    meta_model = RandomForestRegressor(random_state=42)

    feature_columns = meta_dataset.columns.difference([idx_column, class_column])

    meta_model.fit(
        X=meta_dataset.loc[:, feature_columns],
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

    summary_tsfel = ["mean", "std"]
    summary_mfe = ["mean", "sd"]
    tsfel_config = None


out = Parallel(n_jobs=N_JOBS)(
    delayed(task)(i) for i in enumerate(validation_drifting_streams)
)

pd.DataFrame(out).to_csv("{}".format(args.output), index=None)
