from utils import concept_drift
from utils import evaluator
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingAdaptiveTreeClassifier
from experimental_drifts import drifting_streams
import pandas as pd


# How to evaluate
# Drift was detected after a real drift? If yes, how many instances after.
# Delay between drift happening and detection.
# Drift detection when there was no drift (False Alarms).

# What chunk of data are we gonna use? Sliding window in a big stream with multiple drifts
# One "small" data stream with only one drift in multiple positions
# We cannot use a big stream as one meta-instance does not "look real"

# How we are going to adapt the concept drift on the fly? Each X instances we check features?
# Every detected drift we check features?


stream_sizes = 100000
window_size = 500
idx = 0
detected_drifts = []
gen_idx = 0

for g in drifting_streams:
    print("generator {}".format(g.__str__()))
    streamEvaluator = evaluator.Evaluator(windowSize=window_size)
    model = HoeffdingAdaptiveTreeClassifier()
    drift_detector = adwin.ADWIN()
    idx = 0

    for x, y in g.take(stream_sizes):
        y_hat = model.predict_proba_one(x)
        y_predicted = model.predict_one(x)

        drift_detector.update(1 if y == y_predicted else 0)

        streamEvaluator.addResult((x, y), y_hat)
        model.learn_one(x, y)

        if (idx + 1) % window_size == 0:
            print("Accuracy at {}: {}%".format(idx, streamEvaluator.getAccuracy()))

        if drift_detector.drift_detected:
            detected_drifts.append({"g": gen_idx, "idx": idx})

        idx += 1

    gen_idx += 1


df = pd.DataFrame(detected_drifts)
df.to_csv("meta_target.csv", index=False)
