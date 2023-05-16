from river.datasets import synth
from utils import concept_drift
import itertools

META_STREAM_SIZE = 2000

DRIFT_POSITIONS = [500, 1000, 1500]

DRIFT_SPEED = [1, 150, 250, 350]

# TRAIN GENERATORS: AGRAWAL, Mixed, LED, Hyperplane, RandomRBF

# TEST GENERATORS: RandomTree, SEA, Sine, STAGGER, Waveform

# IN REAL-WORLD DATA STREAMS EVALUATE USING A CLASSIFIER THAT RELIES ON A CONCEPT DRIFT DETECTOR
# REAL-WORLD DATASTREAMS: Elec2, CreditCard, Phishing, SMSSPAM, Insects, Spam Assassin, Chess


agrawal_no_drifts = [
    synth.Agrawal(classification_function=i, seed=42) for i in range(0, 9)
]

mixed_no_drifts = [synth.Mixed(classification_function=i, seed=42) for i in range(0, 2)]

LED_no_drifts = [
    synth.LED(seed=42, noise_percentage=i, irrelevant_features=True)
    for i in [0.01, 0.05, 0.10, 0.15]
]

hyperplane_no_drifts = [
    synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i)) for i in range(0, 7)
]

rbf_no_drift = [
    synth.RandomRBF(seed_model=(42 + i), n_centroids=i) for i in range(2, 8)
]


agrawal_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=i, seed=(42 + i)),
        synth.Agrawal(classification_function=(i + 1), seed=(42 + i + 1)),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(itertools.product(range(0, 8), DRIFT_POSITIONS, DRIFT_SPEED))
]

mixed_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Mixed(classification_function=0, seed=i),
        synth.Mixed(classification_function=1, seed=(i + 5)),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(itertools.product(range(42, 46), DRIFT_POSITIONS, DRIFT_SPEED))
]

mixed_drifts = mixed_drifts + [
    concept_drift.ConceptDriftStream(
        synth.Mixed(classification_function=1, seed=i),
        synth.Mixed(classification_function=0, seed=(i + 5)),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(itertools.product(range(46, 50), DRIFT_POSITIONS, DRIFT_SPEED))
]

LED_drifts = [
    concept_drift.ConceptDriftStream(
        synth.LED(seed=42, noise_percentage=i, irrelevant_features=True),
        synth.LED(seed=42, noise_percentage=(i * 2), irrelevant_features=False),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(
        itertools.product([0.01, 0.05, 0.10, 0.15], DRIFT_POSITIONS, DRIFT_SPEED)
    )
]

hyperplane_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i)),
        synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i + 1)),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(itertools.product(range(0, 4), DRIFT_POSITIONS, DRIFT_SPEED))
]

rbf_drift = [
    concept_drift.ConceptDriftStream(
        synth.RandomRBF(seed_model=(42 + i), n_centroids=i),
        synth.RandomRBF(seed_model=(42 + i), n_centroids=i + 2),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(itertools.product(range(2, 6), DRIFT_POSITIONS, DRIFT_SPEED))
]

drifiting_streams = agrawal_no_drifts + agrawal_drifts
