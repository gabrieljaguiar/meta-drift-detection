from river.datasets import synth
from utils import concept_drift
import itertools

META_STREAM_SIZE = 2000

DRIFT_POSITIONS = [500, 1000, 1500]

DRIFT_SPEED = [1, 150, 250, 350]

agrawal_no_drifts = [
    synth.Agrawal(classification_function=i, seed=42) for i in range(0, 9)
]

agrawal_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=i, seed=42),
        synth.Agrawal(classification_function=(i + 1), seed=42),
        width=w,
        position=j,
        size=META_STREAM_SIZE,
    )
    for i, j, w in list(itertools.product(range(0, 8), DRIFT_POSITIONS, DRIFT_SPEED))
]
