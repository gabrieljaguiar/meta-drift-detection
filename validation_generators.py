from river.datasets import synth
from utils import concept_drift
import itertools
import random
from utils.imbalance_generators import BinaryImbalancedStream


META_STREAM_SIZE = 100000

DRIFT_POSITIONS = [20000, 40000, 60000, 80000]

DRIFT_SPEED = [1, 350, 1000]

rng = random.Random(42)


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


validation_drifting_streams = random_tree_drift + rbf_drifts_complex
