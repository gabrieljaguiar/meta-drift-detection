import tsfel
from typing import List, Dict
import pandas as pd
from pymfe.mfe import MFE
import warnings
from tqdm import tqdm
from utils import concept_drift


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


if __name__ == "__main__":
    from train_generators import drifiting_streams, META_STREAM_SIZE

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

    collected_mf = []

    for stream_id, g in tqdm(
        enumerate(drifiting_streams), total=len(drifiting_streams)
    ):
        if isinstance(g, concept_drift.ConceptDriftStream):
            drift_position = g.position
            drift_width = g.width
            stream_name = g.initialStream._repr_content.get("Name")

        else:
            drift_position = 0
            drift_width = 1
            stream_name = g._repr_content.get("Name")

        stream_name = "{}_{}_{}_{}".format(
            stream_id, stream_name, drift_position, drift_width
        )
        data = list(g.take(META_STREAM_SIZE))
        pd_X = pd.DataFrame([data[i][0] for i in range(0, META_STREAM_SIZE)])
        pd_y = pd.DataFrame([data[i][1] for i in range(0, META_STREAM_SIZE)])

        dict_mf = extract_meta_features(
            pd_X,
            pd_y,
            summary_tsfel=["mean", "std"],
            summary_mfe=["mean", "sd"],
            tsfel_config=None,
            mfe_feature_config=mfe_feature_list,
        )

        dict_mf[0]["stream_name"] = stream_name
        collected_mf += dict_mf

    pd_mf = pd.DataFrame(collected_mf)
    pd_mf.fillna(0, inplace=True)
    pd_mf.dropna(axis=1, how="all", inplace=True)

    pd_mf.to_csv("training_meta_features_set_2.csv", index=None)
