from utils import concept_drift
from utils import evaluator
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingAdaptiveTreeClassifier
from experimental_drifts import drifiting_streams, META_STREAM_SIZE
import pandas as pd
from tqdm import tqdm


window_size = 500
idx = 0
meta_dataset = []
grace_period = int(META_STREAM_SIZE * 0.05)
possible_delta_values = [(1 / i) for i in [10, 50, 100, 500, 1000, 5000, 10000]]

range_for_drift = 100


for stream_id, g in tqdm(enumerate(drifiting_streams), total=len(drifiting_streams)):
    # print("generator {}".format(g.__str__()))
    streamEvaluator = evaluator.Evaluator(windowSize=window_size)
    model = HoeffdingAdaptiveTreeClassifier()
    drift_detector = adwin.ADWIN()
    idx = 0

    if isinstance(g, concept_drift.ConceptDriftStream):
        drift_position = g.position
        drift_width = g.width
        stream_name = g.initialStream._repr_content.get("Name")
        stream_name = "{}_{}_{}_{}".format(
            stream_id, stream_name, drift_position, drift_width
        )
        range_for_drift = max(range_for_drift, drift_width)
        g.reset()
    else:
        drift_position = 0
        drift_width = 1
        stream_name = g._repr_content.get("Name")

    number_of_drifts_detected = 0
    distance_to_drift = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for x, y in g.take(META_STREAM_SIZE):
        y_hat = model.predict_proba_one(x)
        y_predicted = model.predict_one(x)

        streamEvaluator.addResult((x, y), y_hat)
        model.learn_one(x, y)
        if idx >= grace_period:
            drift_detector.update(1 if y == y_predicted else 0)

        if drift_detector.drift_detected:
            distance_to_drift += abs(idx - drift_position)
            number_of_drifts_detected += 1
            if (drift_position > 0) and (
                (idx <= drift_position + range_for_drift)
                or (idx >= drift_position - range_for_drift)
            ):
                true_positive += 1
            else:
                false_positive += 1

        idx += 1

    if (drift_position > 0) and (true_positive == 0):
        false_negative += 1

    meta_dataset.append(
        {
            "stream": stream_name,
            "drift_position": drift_position,
            "detection_delay": distance_to_drift,
            "tpr": true_positive,
            "fnr": false_negative,
            "fpr": false_positive,
        }
    )


df = pd.DataFrame(meta_dataset)
df.to_csv("meta_target.csv", index=False)
