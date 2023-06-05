from river.datasets import synth
from utils import concept_drift
import itertools
import random

META_STREAM_SIZE = [2500, 5000, 10000]

DRIFT_POSITIONS = [500, 1000, 1500]

DRIFT_SPEED = [1, 150, 250, 350]

DRIFT_POSITIONS = [1 / 4, 1 / 2, 3 / 4]

rng = random.Random(42)


agrawal_no_drifts = [
    synth.Agrawal(
        classification_function=i, seed=42, balance_classes=True, perturbation=0
    )
    for i in range(0, 9)
]

mixed_no_drift = [
    synth.Mixed(classification_function=i, seed=j)
    for i, j in list(itertools.product(range(0, 2), range(42, 46)))
]  # add seeds to increase number of samples


hyperplane_no_drifts = [
    synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i)) for i in range(0, 7)
]

rbf_no_drift = [
    synth.RandomRBF(seed_model=(42 + i), n_centroids=i) for i in range(2, 8)
]

agrawal_variants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
comb = list(itertools.permutations(agrawal_variants, 2))
variants = rng.sample(comb, 12)


agrawal_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=i[0], seed=(42), balance_classes=True),
        synth.Agrawal(classification_function=i[1], seed=(46), balance_classes=True),
        width=w,
        position=int(size * j),
        size=size,
    )
    for size, i, j, w, in list(
        itertools.product(META_STREAM_SIZE, variants, DRIFT_POSITIONS, DRIFT_SPEED)
    )
]

mixed_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Mixed(classification_function=0, seed=i),
        synth.Mixed(classification_function=1, seed=(i + 5)),
        width=w,
        position=int(size * j),
        size=size,
    )
    for size, i, j, w in list(
        itertools.product(META_STREAM_SIZE, range(42, 46), DRIFT_POSITIONS, DRIFT_SPEED)
    )
]

mixed_drifts = mixed_drifts + [
    concept_drift.ConceptDriftStream(
        synth.Mixed(classification_function=1, seed=i),
        synth.Mixed(classification_function=0, seed=(i + 5)),
        width=w,
        position=int(size * j),
        size=size,
    )
    for size, i, j, w in list(
        itertools.product(META_STREAM_SIZE, range(46, 50), DRIFT_POSITIONS, DRIFT_SPEED)
    )
]


hyperplane_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i)),
        synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i + 1), mag_change=1.0),
        width=w,
        position=int(size * j),
        size=size,
    )
    for size, i, j, w in list(
        itertools.product(META_STREAM_SIZE, range(0, 4), DRIFT_POSITIONS, DRIFT_SPEED)
    )
]

rbf_drift = [
    concept_drift.ConceptDriftStream(
        synth.RandomRBF(seed_model=(42), n_centroids=i),
        synth.RandomRBF(seed_model=(48), n_centroids=i + base_value),
        width=w,
        position=int(size * j),
        size=size,
    )
    for size, i, base_value, j, w in list(
        itertools.product(
            META_STREAM_SIZE, range(2, 6), range(3, 6), DRIFT_POSITIONS, DRIFT_SPEED
        )
    )
]

drifiting_streams = (
    agrawal_no_drifts
    + mixed_no_drift
    + hyperplane_no_drifts
    + rbf_no_drift
    + agrawal_drifts
    + mixed_drifts
    + hyperplane_drifts
    + rbf_drift
)
