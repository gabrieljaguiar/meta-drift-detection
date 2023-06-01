from river.datasets import synth
from utils import concept_drift
from utils.generators.sea import SEAMod
import itertools
import random

# TRAIN GENERATORS: AGRAWAL, Mixed, LED, Hyperplane, RandomRBF

# TEST GENERATORS: RandomTree, SEA, Sine, STAGGER, Waveform

# IN REAL-WORLD DATA STREAMS EVALUATE USING A CLASSIFIER THAT RELIES ON A CONCEPT DRIFT DETECTOR
# REAL-WORLD DATASTREAMS: Elec2, CreditCard, Phishing, SMSSPAM, Insects, Spam Assassin, Chess


# TO TEST WE CAN USE 2000 STREAMS WITH ONE DRIFT AND ALSO USE A BIG STREAM WITH MULTIPLE DRIFTS.


META_STREAM_SIZE = 100000

DRIFT_POSITIONS = [20000, 40000, 60000, 80000]

DRIFT_SPEED = [1, 150, 250, 350]

rng = random.Random(42)


random_tree_no_drift = [
    synth.RandomTree(seed_tree=i, seed_sample=i, max_tree_depth=((i % 2) + 1))
    for i in range(20, 25)
]

sea_no_drift = [SEAMod(variant=i, seed=42, noise=0.0) for i in range(0, 6)]

sine_no_drift = [
    synth.Sine(classification_function=i, seed=42, balance_classes=True)
    for i in range(0, 4)
]

stagger_no_drift = [
    synth.STAGGER(classification_function=i, seed=42) for i in range(0, 3)
]


random_tree_drift = [
    concept_drift.ConceptDriftStream(
        synth.RandomTree(seed_tree=i, seed_sample=i, max_tree_depth=1),
        concept_drift.ConceptDriftStream(
            synth.RandomTree(seed_tree=i * 2, seed_sample=i * 2, max_tree_depth=3),
            concept_drift.ConceptDriftStream(
                synth.RandomTree(seed_tree=i * 3, seed_sample=i * 3, max_tree_depth=6),
                concept_drift.ConceptDriftStream(
                    synth.RandomTree(
                        seed_tree=i * 4, seed_sample=i * 4, max_tree_depth=1
                    ),
                    synth.RandomTree(
                        seed_tree=i * 5, seed_sample=i * 5, max_tree_depth=8
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
    for i, w in list(itertools.product(range(10, 40, 10), DRIFT_SPEED))
]


sea_variants = [0, 1, 2, 3, 4, 5]
comb = list(itertools.permutations(sea_variants, 5))
variants = rng.sample(comb, 6)

sea_drift = [
    concept_drift.ConceptDriftStream(
        SEAMod(variant=v[0], seed=42),
        concept_drift.ConceptDriftStream(
            SEAMod(variant=v[1], seed=42),
            concept_drift.ConceptDriftStream(
                SEAMod(variant=v[2], seed=42),
                concept_drift.ConceptDriftStream(
                    SEAMod(variant=v[3], seed=42),
                    SEAMod(variant=v[4], seed=42),
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
    for v, w in list(itertools.product(variants, DRIFT_SPEED))
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


validation_drifting_streams = (
    random_tree_no_drift
    + random_tree_drift
    + sea_no_drift
    + sea_drift
    + stagger_no_drift
    + stagger_drift
)
