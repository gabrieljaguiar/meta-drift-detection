from river.datasets import synth
from utils import concept_drift
from utils.generators.sea import SEAMod
import itertools
import random
from utils.imbalance_generators import BinaryImbalancedStream


rng = random.Random(42)
sizes = [5000, 10000, 20000, 30000]
DRIFT_SPEED = [1, 350, 1000]


imbalance_ratios = [
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.8, 0.6, 0.4, 0.2, 0.1],
    [0.8, 0.2, 0.8, 0.2, 0.8],
    [0.5, 0.25, 0.1, 0.25, 0.5],
]

agrawal_variants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
comb = list(itertools.permutations(agrawal_variants, 5))
variants = rng.sample(comb, 12)


agrawal_drifts_complex = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(
            synth.Agrawal(classification_function=i[0], seed=(42)),
            imb[0],
        ),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.Agrawal(
                    classification_function=i[1],
                    seed=(42),
                ),
                imb[1],
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.Agrawal(
                        classification_function=i[2],
                        seed=(42),
                    ),
                    imb[2],
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.Agrawal(
                            classification_function=i[3],
                            seed=(42),
                        ),
                        imb[3],
                    ),
                    BinaryImbalancedStream(
                        synth.Agrawal(
                            classification_function=i[4],
                            seed=(42),
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
        itertools.product(imbalance_ratios, sizes, variants, DRIFT_SPEED)
    )
]

hyperplane_drifts_complex = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(
            synth.Hyperplane(seed=(42 + i), n_drift_features=(2 + i)), imb[0]
        ),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.Hyperplane(
                    seed=(42 + i), n_drift_features=(2 + i + 1), mag_change=1.0
                ),
                imb[1],
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.Hyperplane(
                        seed=(42 + i), n_drift_features=(2 + i + 2), mag_change=1.0
                    ),
                    imb[2],
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.Hyperplane(
                            seed=(42 + i), n_drift_features=(2 + i + 3), mag_change=1.0
                        ),
                        imb[3],
                    ),
                    BinaryImbalancedStream(
                        synth.Hyperplane(
                            seed=(42 + i), n_drift_features=(2 + i + 4), mag_change=1.0
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
        itertools.product(imbalance_ratios, sizes, range(0, 4), DRIFT_SPEED)
    )
]

mixed_drifts_complex = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(synth.Mixed(classification_function=1, seed=i), imb[0]),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.Mixed(classification_function=0, seed=i + 5), imb[1]
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.Mixed(classification_function=1, seed=i + 10), imb[2]
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.Mixed(classification_function=1, seed=i + 15), imb[3]
                    ),
                    BinaryImbalancedStream(
                        synth.Mixed(classification_function=0, seed=i + 20), imb[4]
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
        itertools.product(imbalance_ratios, sizes, range(46, 50), DRIFT_SPEED)
    )
]

sea_variants = [0, 1, 2, 3, 4, 5]
comb = list(itertools.permutations(sea_variants, 5))
variants = rng.sample(comb, 3)

sea_drift_complex = [
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

sine_variants = [0, 1, 2, 3]
comb = list(itertools.combinations_with_replacement(sine_variants, 5))
variants = rng.sample(comb, 5)

sine_drift_complex = [
    concept_drift.ConceptDriftStream(
        BinaryImbalancedStream(
            synth.Sine(classification_function=var[0], seed=42), imb[0]
        ),
        concept_drift.ConceptDriftStream(
            BinaryImbalancedStream(
                synth.Sine(classification_function=var[1], seed=42), imb[1]
            ),
            concept_drift.ConceptDriftStream(
                BinaryImbalancedStream(
                    synth.Sine(classification_function=var[2], seed=42), imb[2]
                ),
                concept_drift.ConceptDriftStream(
                    BinaryImbalancedStream(
                        synth.Sine(classification_function=var[3], seed=42), imb[3]
                    ),
                    BinaryImbalancedStream(
                        synth.Sine(classification_function=var[4], seed=42), imb[4]
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
    for imb, var, s, w in list(
        itertools.product(imbalance_ratios, variants, sizes, DRIFT_SPEED)
    )
]


meta_data_streams = (
    agrawal_drifts_complex
    + hyperplane_drifts_complex
    + mixed_drifts_complex
    + sea_drift_complex
    + sine_drift_complex
)
