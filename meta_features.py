import tsfel
from typing import List, Dict
import random
import pandas as pd
import numpy as np
from pymfe.mfe import MFE
import warnings

# Features
# abslute energy ok
# total energy ok
# centroid ok
# entropy ok
# Area under the curve
# Average number of elements by class
# Percentage of elements of the minority class
# Percentage of elements of the majority class
# Number of classes
# Number of attributes
# Number of numeric attributes
# Number of nominal (symbolic) attribute

# Fisher's discriminant ratio
# numerator <- function(j, data) {

#  tmp <- branch(data, j)
#  aux <- nrow(tmp) * (colMeans(tmp) -
#    colMeans(data[,-ncol(data), drop=FALSE]))^2
#  return(aux)
# }

# denominator <- function(j, data) {

#  tmp <- branch(data, j)
#  aux <- rowSums((t(tmp) - colMeans(tmp))^2)
#  return(aux)
# }


def extract_meta_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    summary: List = None,
    tsfel_config: Dict = None,
    mfe_feature_config: List = None,
) -> Dict:
    if tsfel_config is None:
        domain = tsfel.get_features_by_domain()
        cfg = {}
        cfg["temporal"] = domain.get("temporal")
    else:
        cfg = tsfel_config

    if mfe_feature_config == None:
        mfe_feature_list = "all"
    else:
        mfe_feature_list = mfe_feature_config

    if summary is None:
        summary = ["max", "min", "mean", "var"]

    df_mfe_features, df_tsfel_features = None, None

    if cfg != {}:
        fs = X.shape[0] * 0.2
        tsfel_features = tsfel.calc_window_features(cfg, X, fs=fs)
        summarized = (
            tsfel_features.T.groupby(lambda x: x.split("_", 1)[1]).agg(summary).T
        )
        flat = summarized.unstack().sort_index(level=1)
        flat.columns = flat.columns.map("_".join)

        df_tsfel_features = pd.DataFrame(flat)

    if mfe_feature_list != []:
        print("f1" in MFE.valid_metafeatures())
        mfe_extractor = MFE(features=mfe_feature_list, summary=summary, groups=["all"])
        mfe_extractor.fit(X.to_numpy(), y.to_numpy())
        mfe_features = mfe_extractor.extract(verbose=1, suppress_warnings=True)
        df_mfe_features = pd.DataFrame(mfe_features[1:], columns=mfe_features[0])

    meta_features = pd.concat([df_mfe_features, df_tsfel_features], axis=1)

    return meta_features.to_dict("records")


if __name__ == "__main__":
    from train_generators import agrawal_no_drifts, META_STREAM_SIZE

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # agrawal_no_drifts = agrawal_no_drifts[0:1]

    meta_database = []

    for stream in agrawal_no_drifts:
        X_array = []
        y_array = []

        for X, y in stream.take(META_STREAM_SIZE):
            X_array.append(X)
            y_array.append(y)

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

        df_X = pd.DataFrame(X_array)
        df_y = pd.DataFrame(y_array)

        mf = extract_meta_features(
            df_X,
            df_y,
            summary=["mean", "var"],
            mfe_feature_config=mfe_feature_list,
            tsfel_config={},
        )

        print(mf)

        meta_database += mf

    df_meta_db = pd.DataFrame(meta_database)

    df_meta_db.to_csv("mf.csv", index=None)
