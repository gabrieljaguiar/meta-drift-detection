from river.datasets import synth
from utils import concept_drift
from utils.generators.sea import SEAMod
from utils.generators.rbf import RBFMod
import itertools
import random
from utils.imbalance_generators import BinaryImbalancedStream


# TRAIN GENERATORS: AGRAWAL, Mixed, LED, Hyperplane, RandomRBF

# TEST GENERATORS: RandomTree, SEA, Sine, STAGGER, Waveform

# IN REAL-WORLD DATA STREAMS EVALUATE USING A CLASSIFIER THAT RELIES ON A CONCEPT DRIFT DETECTOR
# REAL-WORLD DATASTREAMS: Elec2, CreditCard, Phishing, SMSSPAM, Insects, Spam Assassin, Chess


# TO TEST WE CAN USE 2000 STREAMS WITH ONE DRIFT AND ALSO USE A BIG STREAM WITH MULTIPLE DRIFTS.


META_STREAM_SIZE = 100000

DRIFT_POSITIONS = [20000, 40000, 60000, 80000]

DRIFT_SPEED = [1, 350, 1000]

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

sizes = [10000, 20000, 30000]
sizes = [20000, 30000, 50000]

imbalance_ratios = [
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.8, 0.4, 0.3, 0.2, 0.1],
    [0.8, 0.2, 0.8, 0.2, 0.8],
    [0.5, 0.25, 0.1, 0.25, 0.5],
]

random_tree_drift = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(
            synth.RandomTree(seed_tree=i, seed_sample=i, max_tree_depth=1), imb[0]
        ),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.RandomTree(seed_tree=i, seed_sample=i * 2, max_tree_depth=3),
                imb[1],
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.RandomTree(seed_tree=i, seed_sample=i * 3, max_tree_depth=6),
                    imb[2],
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.RandomTree(
                            seed_tree=i, seed_sample=i * 4, max_tree_depth=1
                        ),
                        imb[3],
                    ),
                    BinaryImbalancedStream(
                        synth.RandomTree(
                            seed_tree=i, seed_sample=i * 5, max_tree_depth=8
                        ),
                        imb[4],
                    ),
                    position=s / 5,
                    width=w,
                    size=s,
                ),
                position=s / 5,
                width=w,
                size=s,
            ),
            position=s / 5,
            width=w,
            size=s,
        ),
        position=s / 5,
        width=w,
        size=s,
    )
    for imb, s, i, w in list(
        itertools.product(imbalance_ratios, sizes, range(10, 40, 10), DRIFT_SPEED)
    )
]


sea_variants = [0, 1, 2, 3, 4, 5]
comb = list(itertools.permutations(sea_variants, 5))
variants = rng.sample(comb, 3)


sea_drift = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(SEAMod(variant=v[0], seed=42), imb[0]),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(SEAMod(variant=v[1], seed=42), imb[1]),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(SEAMod(variant=v[2], seed=42), imb[2]),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(SEAMod(variant=v[3], seed=42), imb[3]),
                    BinaryImbalancedStream(SEAMod(variant=v[4], seed=42), imb[4]),
                    position=s / 5,
                    width=w,
                    size=s,
                ),
                position=s / 5,
                width=w,
                size=s,
            ),
            position=s / 5,
            width=w,
            size=s,
        ),
        position=s / 5,
        width=w,
        size=s,
    )
    for imb, s, v, w in list(
        itertools.product(imbalance_ratios, sizes, variants, DRIFT_SPEED)
    )
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
                    position=s / 5,
                    width=w,
                    size=s,
                ),
                position=s / 5,
                width=w,
                size=s,
            ),
            position=s / 5,
            width=w,
            size=s,
        ),
        position=s / 5,
        width=w,
        size=s,
    )
    for s, w in list(itertools.product(sizes, DRIFT_SPEED))
]

stagger_drift = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(
            synth.STAGGER(classification_function=0, seed=42), imb[0]
        ),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.STAGGER(classification_function=1, seed=42), imb[1]
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.STAGGER(classification_function=2, seed=42), imb[2]
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.STAGGER(classification_function=0, seed=42), imb[3]
                    ),
                    BinaryImbalancedStream(
                        synth.STAGGER(classification_function=1, seed=42), imb[4]
                    ),
                    position=s / 5,
                    width=w,
                    size=s,
                ),
                position=s / 5,
                width=w,
                size=s,
            ),
            position=s / 5,
            width=w,
            size=s,
        ),
        position=s / 5,
        width=w,
        size=s,
    )
    for imb, s, w in list(itertools.product(imbalance_ratios, sizes, DRIFT_SPEED))
]


n_centroids = [12, 20, 32, 40, 60]
comb = list(itertools.permutations(n_centroids, 5))
# print(comb)
centroids = rng.sample(comb, 3)


rbf_drifts_complex = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(
            synth.RandomRBFDrift(
                seed_model=(42),
                seed_sample=42,
                n_centroids=c[0],
                n_drift_centroids=int(c[0] / 2),
            ),
            imb[0],
        ),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.RandomRBFDrift(
                    seed_model=(12),
                    seed_sample=12,
                    n_centroids=c[0],
                    n_drift_centroids=int(c[0] / 2),
                ),
                imb[1],
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.RandomRBFDrift(
                        seed_model=(24),
                        seed_sample=24,
                        n_centroids=c[0],
                        n_drift_centroids=int(c[0] / 2),
                    ),
                    imb[2],
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.RandomRBFDrift(
                            seed_model=(36),
                            seed_sample=36,
                            n_centroids=c[0],
                            n_drift_centroids=int(c[0] / 2),
                        ),
                        imb[3],
                    ),
                    BinaryImbalancedStream(
                        synth.RandomRBFDrift(
                            seed_model=(48),
                            seed_sample=48,
                            n_centroids=c[0],
                            n_drift_centroids=int(c[0] / 2),
                        ),
                        imb[4],
                    ),
                    position=s / 5,
                    width=w,
                    size=s,
                ),
                position=s / 5,
                width=w,
                size=s,
            ),
            position=s / 5,
            width=w,
            size=s,
        ),
        position=s / 5,
        width=w,
        size=s,
    )
    for imb, s, c, w in list(
        itertools.product(imbalance_ratios, sizes, centroids, DRIFT_SPEED)
    )
]


validation_drifting_streams = (
    # random_tree_no_drift
    random_tree_drift
    + rbf_drifts_complex
    # + sea_no_drift
    # + sea_drift
    # + stagger_no_drift
    # + stagger_drift
)
