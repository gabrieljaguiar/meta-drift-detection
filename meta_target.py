from utils import concept_drift
from utils import evaluator
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingAdaptiveTreeClassifier

sudden_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=0, seed=42),
        synth.Agrawal(classification_function=8, seed=42),
        width=1,
        position=10000,
        angle=0,
    ),
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=2, seed=42),
        synth.Agrawal(classification_function=4, seed=42),
        width=1,
        position=10000,
        angle=0,
    ),
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=7, seed=42),
        synth.Agrawal(classification_function=2, seed=42),
        width=1,
        position=10000,
        angle=0,
    ),
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=6, seed=42),
        synth.Agrawal(classification_function=1, seed=42),
        width=1,
        position=10000,
        angle=0,
    ),
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=2, seed=42),
        synth.Agrawal(classification_function=5, seed=42),
        width=1,
        position=10000,
        angle=0,
    ),
]


# How to evaluate
# Drift was detected after a real drift? If yes, how many instances after.
# Delay between drift happening and detection.
# Drift detection when there was no drift (False Alarms).


stream_sizes = 20000
window_size = 500
idx = 0

for g in sudden_drifts:
    print("generator {}".format(g.__str__()))
    streamEvaluator = evaluator.Evaluator(windowSize=window_size)
    model = HoeffdingAdaptiveTreeClassifier()
    drift_detector = adwin.ADWIN()
    idx = 0
    detected_drifts = []

    for x, y in g.take(stream_sizes):
        y_hat = model.predict_proba_one(x)
        y_predicted = model.predict_one(x)

        drift_detector.update(1 if y == y_predicted else 0)

        streamEvaluator.addResult((x, y), y_hat)
        model.learn_one(x, y)

        if drift_detector.drift_detected:
            detected_drifts.append({"idx": idx})

        idx += 1
