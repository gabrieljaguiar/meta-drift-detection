from tqdm import tqdm
from pymfe.mfe import MFE
from utils import concept_drift
from typing import List, Dict
from joblib import Parallel, delayed

import warnings
import argparse
import tsfel
import pandas as pd
import itertools

IMBALANCE_SCENARIO = imbalance_ratios = {
    "0.5_0.5_0.5_0.5_0.5": "STABLE",
    "0.8_0.4_0.3_0.2_0.1": "DECREASING",
    "0.8_0.2_0.8_0.2_0.8": "FLIPPING",
    "0.5_0.25_0.1_0.25_0.5": "INCREASE_DECREASE",
}


def extract_meta_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    summary_tsfel: List = None,
    tsfel_config: Dict = None,
    summary_mfe: List = None,
    mfe_feature_config: List = None,
) -> Dict:
    if tsfel_config is None:
        domain = tsfel.get_features_by_domain()
        cfg = {}
        cfg["temporal"] = domain.get("temporal")
        cfg["statistical"] = domain.get("statistical")
    else:
        cfg = tsfel_config

    if mfe_feature_config == None:
        mfe_feature_list = "all"
    else:
        mfe_feature_list = mfe_feature_config

    if summary_mfe is None:
        summary_mfe = ["max", "min", "mean", "var"]
    if summary_tsfel is None:
        summary_tsfel = ["max", "min", "mean", "var"]

    df_mfe_features, df_tsfel_features = None, None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        if cfg != {}:
            fs = X.shape[0] * 0.2
            tsfel_features = tsfel.calc_window_features(cfg, X, fs=fs)
            summarized = (
                tsfel_features.T.groupby(lambda x: x.split("_", 1)[1])
                .agg(summary_tsfel)
                .T
            )
            flat = summarized.unstack().sort_index(level=1)
            flat.columns = flat.columns.map("_".join)

            df_tsfel_features = pd.DataFrame(flat)

        if mfe_feature_list != []:
            mfe_extractor = MFE(
                features=mfe_feature_list, summary=summary_mfe, groups=["all"]
            )
            mfe_extractor.fit(X.to_numpy(), y.to_numpy())
            mfe_features = mfe_extractor.extract(verbose=0, suppress_warnings=True)
            df_mfe_features = pd.DataFrame(mfe_features[1:], columns=mfe_features[0])

        meta_features = pd.concat([df_mfe_features, df_tsfel_features], axis=1)

    return meta_features.to_dict("records")


def task(arg):
    global META_WINDOW_SIZE, SET_GROUP
    stream_id, g = arg
    sizes = []

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

    meta_samples = []

    data = list(g.take(size))
    pd_X = pd.DataFrame([data[i][0] for i in range(0, size)])
    pd_y = pd.DataFrame([data[i][1] for i in range(0, size)])

    if SET_GROUP == 1:
        tsfel_cfg = {}
        mfe_cfg = mfe_feature_list
    if SET_GROUP == 2:
        tsfel_cfg = None
        mfe_cfg = []
    if SET_GROUP == 3:
        tsfel_cfg = None
        mfe_cfg = mfe_feature_list

    print("Extracting {}".format(stream_name))

    dict_mf = extract_meta_features(
        pd_X,
        pd_y,
        summary_tsfel=["mean", "max", "min"],
        summary_mfe=["mean", "sd"],
        tsfel_config=tsfel_cfg,
        mfe_feature_config=mfe_cfg,
    )

    dict_mf[0]["stream_name"] = stream_name
    meta_samples.append(dict_mf[0])
    return meta_samples


if __name__ == "__main__":
    from meta_data_generators import meta_data_streams

    parser = argparse.ArgumentParser(description="Meta-feature extraction")

    parser.add_argument(
        "--output", type=str, help="output file", default="meta_features.csv"
    )

    parser.add_argument(
        "--feature-set",
        type=int,
        help="Set of features to be used. 1 = MFE, 2 = TSFEL, 3 = BOTH",
        default=1,
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of multiple process"
    )

    args = parser.parse_args()

    N_JOBS = args.n_jobs
    SET_GROUP = args.feature_set

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

    collected_mf = []

    collected_mf = Parallel(n_jobs=N_JOBS)(
        delayed(task)(i) for i in enumerate(meta_data_streams)
    )

    collected_mf = list(itertools.chain.from_iterable(collected_mf))

    pd_mf = pd.DataFrame(collected_mf)
    pd_mf.fillna(0, inplace=True)
    pd_mf.dropna(axis=1, how="all", inplace=True)

    pd_mf.to_csv("{}".format(args.output), index=None)
