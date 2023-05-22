from river.datasets import synth
from utils import concept_drift
import itertools

# TRAIN GENERATORS: AGRAWAL, Mixed, LED, Hyperplane, RandomRBF

# TEST GENERATORS: RandomTree, SEA, Sine, STAGGER, Waveform

# IN REAL-WORLD DATA STREAMS EVALUATE USING A CLASSIFIER THAT RELIES ON A CONCEPT DRIFT DETECTOR
# REAL-WORLD DATASTREAMS: Elec2, CreditCard, Phishing, SMSSPAM, Insects, Spam Assassin, Chess


# TO TEST WE CAN USE 2000 STREAMS WITH ONE DRIFT AND ALSO USE A BIG STREAM WITH MULTIPLE DRIFTS.


META_STREAM_SIZE = 100000

DRIFT_POSITIONS = [20000, 40000, 60000, 80000]

DRIFT_SPEED = [1, 150, 250, 350]


random_tree_no_drift = [
    synth.RandomTree(seed_tree=i, seed_sample=i) for i in range(20, 25)
]

sea_no_drift = [synth.SEA(variant=i, seed=42) for i in range(0, 3)]

sine_no_drift = [
    synth.Sine(classification_function=i, seed=42, balance_classes=True)
    for i in range(0, 4)
]

stagger_no_drift = [
    synth.STAGGER(classification_function=i, seed=42) for i in range(0, 3)
]

waveform_no_drift = [synth.Waveform(seed=i) for i in range(10, 40, 5)]


random_tree_drift = [
    concept_drift.ConceptDriftStream(
        synth.RandomTree(seed_tree=i, seed_sample=i),
        concept_drift.ConceptDriftStream(
            synth.RandomTree(seed_tree=i * 2, seed_sample=i * 2),
            concept_drift.ConceptDriftStream(
                synth.RandomTree(seed_tree=i * 3, seed_sample=i * 3),
                concept_drift.ConceptDriftStream(
                    synth.RandomTree(seed_tree=i * 4, seed_sample=i * 4),
                    synth.RandomTree(seed_tree=i * 5, seed_sample=i * 5),
                    position=20000,
                    width=w,
                    size=META_STREAM_SIZE,
                ),
                position=20000,
                width=w,
                size=META_STREAM_SIZE,
            ),
            position=20000,
            width=w,
            size=META_STREAM_SIZE,
        ),
        position=20000,
        width=w,
        size=META_STREAM_SIZE,
    )
    for i, w in list(itertools.product(range(10, 40, 10), DRIFT_SPEED))
]

sea_drift = [
    concept_drift.ConceptDriftStream(
        synth.SEA(variant=0, seed=42),
        concept_drift.ConceptDriftStream(
            synth.SEA(variant=1, seed=42),
            concept_drift.ConceptDriftStream(
                synth.SEA(variant=2, seed=42),
                concept_drift.ConceptDriftStream(
                    synth.SEA(variant=0, seed=42),
                    synth.SEA(variant=1, seed=42),
                    position=20000,
                    width=w,
                    size=META_STREAM_SIZE,
                ),
                position=20000,
                width=w,
                size=META_STREAM_SIZE,
            ),
            position=20000,
            width=w,
            size=META_STREAM_SIZE,
        ),
        position=20000,
        width=w,
        size=META_STREAM_SIZE,
    )
    for w in DRIFT_SPEED
]


sine_drift = [
    concept_drift.ConceptDriftStream(
        synth.Sine(classification_function=0, seed=42, balance_classes=True),
        concept_drift.ConceptDriftStream(
            synth.Sine(classification_function=1, seed=42, balance_classes=True),
            concept_drift.ConceptDriftStream(
                synth.Sine(classification_function=2, seed=42, balance_classes=True),
                concept_drift.ConceptDriftStream(
                    synth.Sine(
                        classification_function=3, seed=42, balance_classes=True
                    ),
                    synth.Sine(
                        classification_function=1, seed=42, balance_classes=True
                    ),
                    position=20000,
                    width=w,
                    size=META_STREAM_SIZE,
                ),
                position=20000,
                width=w,
                size=META_STREAM_SIZE,
            ),
            position=20000,
            width=w,
            size=META_STREAM_SIZE,
        ),
        position=20000,
        width=w,
        size=META_STREAM_SIZE,
    )
    for w in DRIFT_SPEED
]

stagger_drift = [
    concept_drift.ConceptDriftStream(
        synth.STAGGER(classification_function=0, seed=42),
        concept_drift.ConceptDriftStream(
            synth.STAGGER(classification_function=1, seed=42),
            concept_drift.ConceptDriftStream(
                synth.STAGGER(classification_function=2, seed=42),
                concept_drift.ConceptDriftStream(
                    synth.STAGGER(classification_function=0, seed=42),
                    synth.STAGGER(classification_function=1, seed=42),
                    position=20000,
                    width=w,
                    size=META_STREAM_SIZE,
                ),
                position=20000,
                width=w,
                size=META_STREAM_SIZE,
            ),
            position=20000,
            width=w,
            size=META_STREAM_SIZE,
        ),
        position=20000,
        width=w,
        size=META_STREAM_SIZE,
    )
    for w in DRIFT_SPEED
]


waveform_drift = [
    concept_drift.ConceptDriftStream(
        synth.Waveform(seed=i),
        concept_drift.ConceptDriftStream(
            synth.Waveform(seed=i * 2),
            concept_drift.ConceptDriftStream(
                synth.Waveform(seed=i * 3),
                concept_drift.ConceptDriftStream(
                    synth.Waveform(seed=i * 4),
                    synth.Waveform(seed=i * 5),
                    position=20000,
                    width=w,
                    size=META_STREAM_SIZE,
                ),
                position=20000,
                width=w,
                size=META_STREAM_SIZE,
            ),
            position=20000,
            width=w,
            size=META_STREAM_SIZE,
        ),
        position=20000,
        width=w,
        size=META_STREAM_SIZE,
    )
    for i, w in list(itertools.product(range(10, 40, 10), DRIFT_SPEED))
]


validation_drifting_streams = (
    random_tree_no_drift
    + random_tree_drift
    + sea_no_drift
    + sea_drift
    + stagger_no_drift
    + stagger_drift
    + waveform_no_drift
    + waveform_drift
)
