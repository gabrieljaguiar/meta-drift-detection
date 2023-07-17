from validation_generators import validation_drifting_streams
from utils import concept_drift
import pandas as pd


IMBALANCE_SCENARIO = imbalance_ratios = {
    "0.5_0.5_0.5_0.5_0.5": "STABLE",
    "0.8_0.4_0.3_0.2_0.1": "INVERTED",
    "0.8_0.2_0.8_0.2_0.8": "FLIPPING",
    "0.5_0.25_0.1_0.25_0.5": "INCREASE_DECREASE",
}


for stream_id, g in enumerate(validation_drifting_streams):
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

    # adwin_df = pd.read_csv("./metrics/ADWIN_{}.csv".format(stream_name))
    # adwin_df["model"] = "ADWIN"
    # kswin_df = pd.read_csv("./metrics/KSWIN_{}.csv".format(stream_name))
    # kswin_df["model"] = "KSWIN"
    meta_df = pd.read_csv("./metrics/META_2_5000_5000_{}.csv".format(stream_name))
    meta_df["model"] = "META"
    # hddm_df = pd.read_csv("./metrics/HDDM_{}.csv".format(stream_name))
    # hddm_df["model"] = "HDDM"
    ddm_df = pd.read_csv("./metrics/DDM_{}.csv".format(stream_name))
    ddm_df["model"] = "DDM"
    random_df = pd.read_csv("./metrics/RANDOM_2_5000_5000_{}.csv".format(stream_name))
    random_df["model"] = "RANDOM"

    combined_df = pd.concat([meta_df, random_df, ddm_df], axis=0)

    combined_df.drop(["dd"], axis=1, inplace=True)

    combined_df["rank"] = combined_df.groupby("idx")["gmean"].rank(
        ascending=False, method="first"
    )

    combined_df.groupby("model")["rank"].value_counts().to_csv(
        "./ranks/{}.csv".format(stream_name)
    )

    # print(combined_df)
