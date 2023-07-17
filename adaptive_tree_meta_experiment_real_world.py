from joblib import Parallel, delayed
from utils import concept_drift
from utils import adaptiveADWIN
from utils.queue import Queue
from utils.evaluator import Evaluator
from river import naive_bayes
from meta_features import extract_meta_features
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.preprocessing import StandardScaler
from river.drift import KSWIN, adwin, binary
from river.tree import HoeffdingAdaptiveTreeClassifier
from river.stream import iter_arff
import os
import pandas as pd
import numpy as np
import argparse
import multiprocessing
from tqdm import tqdm
from glob import glob
from utils.arff import ArffStream
import random

parser = argparse.ArgumentParser(description="Meta-Drift evaluation")


parser.add_argument("--model", type=str, help="FIXED or META", default="META")
parser.add_argument("--dd", type=str, help="FIXED or META", default="KSWIN")

parser.add_argument(
    "--mf",
    type=str,
    help="Meta-feature file to be used",
    default="training_meta_features_set_1.csv",
)

parser.add_argument(
    "--feature-set",
    type=int,
    help="Set of features to be used. 1 = MFE, 2 = TSFEL, 3 = BOTH",
    default=1,
)


parser.add_argument("--output", type=str, help="results_file", default="results")

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

META_WINDOW_SIZE = args.mt
DD_MODEL = args.dd
MODEL = args.model
STRIDE_WINDOW = args.st
N_JOBS = multiprocessing.cpu_count() if args.n_jobs == -1 else args.n_jobs
SET_GROUP = args.feature_set

rng = random.Random(42)

IMBALANCE_SCENARIO = imbalance_ratios = {
    "0.5_0.5_0.5_0.5_0.5": "STABLE",
    "0.8_0.4_0.3_0.2_0.1": "DECREASING",
    "0.8_0.2_0.8_0.2_0.8": "FLIPPING",
    "0.5_0.25_0.1_0.25_0.5": "INCREASE_DECREASE",
}

print("Starting experiment")
print("OUTPUT FILE: {}".format(args.output))
print("META-WINDOW: {}".format(META_WINDOW_SIZE))
print("STRIDE_WINDOW: {}".format(STRIDE_WINDOW))
print("PARALLALEL JOBS: {}".format(N_JOBS))


def getFeatures(chunk):
    global feature_columns
    x_queue, y_queue = chunk

    mfe_feature_list = [
        "joint_ent",
        "ns_ratio",
        "can_cor",
        "gravity",
        "kurtosis",
        "skewness",
        "sparsity",
        "sd_ratio",
        "class_ent",
        "class_conc",
        "class_ent",
        "nr_cor_attr",
        "c2",
        "t4",
        "f1",
        "f1v",
        "f2",
        "f3",
        "f4",
        "n1",
        "n2",
    ]
    if SET_GROUP == 1:
        tsfel_cfg = {}
        mfe_cfg = mfe_feature_list
    if SET_GROUP == 2:
        tsfel_cfg = None
        mfe_cfg = []
    if SET_GROUP == 3:
        tsfel_cfg = None
        mfe_cfg = mfe_feature_list

    pd_X = pd.DataFrame(x_queue.queue)
    pd_y = pd.DataFrame(y_queue.queue)

    dict_mf = extract_meta_features(
        pd_X,
        pd_y,
        summary_tsfel=["mean", "max", "min"],
        summary_mfe=["mean", "sd"],
        tsfel_config=tsfel_cfg,
        mfe_feature_config=mfe_cfg,
    )

    meta_features_df = pd.DataFrame(dict_mf)
    meta_features_df.fillna(0, inplace=True)
    meta_features_df = meta_features_df.loc[:, feature_columns]

    return meta_features_df


def task(arg):
    global META_WINDOW_SIZE, META_STREAM_SIZE, STRIDE_WINDOW, DD_MODEL, rng
    _, (stream_name, g) = arg
    g = ArffStream(g)

    model_adwin = HoeffdingAdaptiveTreeClassifier(drift_detector=adwin.ADWIN(), seed=42)

    model_hddm = HoeffdingAdaptiveTreeClassifier(
        drift_detector=binary.HDDM_W(), seed=42
    )

    model_kswin = HoeffdingAdaptiveTreeClassifier(
        drift_detector=KSWIN(seed=42), seed=42
    )

    model_ddm = HoeffdingAdaptiveTreeClassifier(drift_detector=binary.DDM(), seed=42)

    for c in g.target_range:
        model_adwin.classes.add(c)
        model_hddm.classes.add(c)
        model_kswin.classes.add(c)
        model_ddm.classes.add(c)

    stride = 0

    chunk_idx = 0

    grace_period = 100

    print("Evaluating {} with {}".format(stream_name, MODEL))

    stride = STRIDE_WINDOW

    X_queue = Queue(META_WINDOW_SIZE)
    y_queue = Queue(META_WINDOW_SIZE)

    evaluator = Evaluator(500)

    metrics = []

    DD_MODELS = ["ADWIN", "KSWIN", "DDM", "HDDM"]

    for idx, (x, y) in enumerate(g):
        X_queue.insert(x)
        y_queue.insert(y)
        stride += 1

        if (
            X_queue.getNumberOfElements() == META_WINDOW_SIZE
            and stride >= STRIDE_WINDOW
        ):
            # print("Lets extract features")

            if MODEL == "META":
                meta_features = getFeatures((X_queue, y_queue))

                # meta_features_df = pd.DataFrame(meta_features)
                # meta_features_df.fillna(0, inplace=True)
                rankings = [
                    meta_model_adwin.predict(meta_features),
                    meta_model_kswin.predict(meta_features),
                    meta_model_ddm.predict(meta_features),
                    meta_model_hddm.predict(meta_features),
                ]

                DD_MODEL = DD_MODELS[rankings.index(min(rankings))]
                stride = 0

            elif MODEL == "RANDOM":
                DD_MODEL = DD_MODELS[rng.choice([0, 1, 2, 3])]

                stride = 0

        if idx > grace_period:
            if DD_MODEL == "ADWIN":
                evaluator.addResult((x, y), model_adwin.predict_proba_one(x))
            if DD_MODEL == "KSWIN":
                evaluator.addResult((x, y), model_kswin.predict_proba_one(x))
            if DD_MODEL == "DDM":
                evaluator.addResult((x, y), model_ddm.predict_proba_one(x))
            if DD_MODEL == "HDDM":
                evaluator.addResult((x, y), model_hddm.predict_proba_one(x))

        model_adwin.learn_one(x, y)
        model_ddm.learn_one(x, y)
        model_hddm.learn_one(x, y)
        model_kswin.learn_one(x, y)

        if (idx + 1) % 500 == 0:
            metrics.append(
                {
                    "idx": idx,
                    "dd": DD_MODEL,
                    "acc": evaluator.getAccuracy(),
                    "gmean": evaluator.getGMean(),
                }
            )

    metrics_df = pd.DataFrame(metrics)
    if MODEL == "META":
        metrics_df.to_csv("./metrics/REAL_{}_{}.csv".format(MODEL, stream_name))
    else:
        metrics_df.to_csv(
            "./metrics/REAL_{}_{}_{}.csv".format(MODEL, DD_MODEL, stream_name)
        )

    item = {
        "stream": stream_name,
        "chunk": chunk_idx,
        "idx": idx,
        "acc": metrics_df["acc"].mean(),
        "gmean": metrics_df["gmean"].mean(),
    }

    print("Finished evaluating {} with {}".format(stream_name, "META"))

    # print(meta_target_df)

    return item


if __name__ == "__main__":
    from validation_generators import validation_drifting_streams

    if MODEL == "META":
        print("META_FEATURE FILE: {}".format(args.mf))

        meta_target_df = pd.read_csv("meta_target.csv")

        meta_target_filtered = meta_target_df.loc[
            meta_target_df.groupby("stream").gmean.idxmax()
        ].reset_index(drop=True)

        meta_target_df["rank"] = meta_target_df.groupby("stream")["gmean"].rank()

        training_meta_features = pd.read_csv("./{}".format(args.mf))

        training_meta_features = training_meta_features.fillna(0)

        meta_target = meta_target_df.loc[:, ["stream", "model", "rank"]]

        meta_dataset = training_meta_features.merge(
            right=meta_target, how="left", left_on="stream_name", right_on="stream"
        )

        meta_dataset_hddm = meta_dataset[meta_dataset["model"] == "HDDM"]

        meta_dataset_ddm = meta_dataset[meta_dataset["model"] == "DDM"]

        meta_dataset_adwin = meta_dataset[meta_dataset["model"] == "ADWIN"]

        meta_dataset_kswin = meta_dataset[meta_dataset["model"] == "KSWIN"]

        idx_column = "stream"
        class_column = "rank"

        meta_model_hddm = Pipeline([("scaler", StandardScaler()), ("rf", SVR())])
        meta_model_ddm = Pipeline([("scaler", StandardScaler()), ("rf", SVR())])
        meta_model_adwin = Pipeline([("scaler", StandardScaler()), ("rf", SVR())])
        meta_model_kswin = Pipeline([("scaler", StandardScaler()), ("rf", SVR())])
        # meta_model = DecisionTreeClassifier()

        meta_dataset.drop(["stream_name", "model"], axis=1, inplace=True)
        feature_columns = meta_dataset.columns.difference([idx_column, class_column])

        meta_model_hddm.fit(
            X=meta_dataset_hddm.loc[:, feature_columns],
            y=meta_dataset_hddm.loc[:, class_column],
        )

        meta_model_ddm.fit(
            X=meta_dataset_ddm.loc[:, feature_columns],
            y=meta_dataset_ddm.loc[:, class_column],
        )

        meta_model_adwin.fit(
            X=meta_dataset_adwin.loc[:, feature_columns],
            y=meta_dataset_adwin.loc[:, class_column],
        )

        meta_model_kswin.fit(
            X=meta_dataset_kswin.loc[:, feature_columns],
            y=meta_dataset_kswin.loc[:, class_column],
        )

    datasets = glob("./datasets/imbalanced/*.arff")

    complex_drifts = [(os.path.basename(ds), ds) for ds in datasets]

    out = Parallel(n_jobs=N_JOBS)(delayed(task)(i) for i in enumerate(complex_drifts))

    pd.DataFrame(out).to_csv("{}".format(args.output), index=None)
