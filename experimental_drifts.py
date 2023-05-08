from river.datasets import synth
from utils import concept_drift

sudden_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=0, seed=42),
        concept_drift.ConceptDriftStream(
            synth.Agrawal(classification_function=1, seed=42),
            concept_drift.ConceptDriftStream(
                synth.Agrawal(classification_function=2, seed=42),
                concept_drift.ConceptDriftStream(
                    synth.Agrawal(classification_function=3, seed=42),
                    synth.Agrawal(classification_function=4, seed=42),
                    width=1,
                    position=20000,
                    angle=0,
                ),
                width=1,
                position=20000,
                angle=0,
            ),
            width=1,
            position=20000,
            angle=0,
        ),
        width=1,
        position=20000,
        angle=0,
    ),
]


gradual_drifts = [
    concept_drift.ConceptDriftStream(
        synth.Agrawal(classification_function=0, seed=42),
        concept_drift.ConceptDriftStream(
            synth.Agrawal(classification_function=1, seed=42),
            concept_drift.ConceptDriftStream(
                synth.Agrawal(classification_function=2, seed=42),
                concept_drift.ConceptDriftStream(
                    synth.Agrawal(classification_function=3, seed=42),
                    synth.Agrawal(classification_function=4, seed=42),
                    width=500,
                    position=10000,
                    angle=0,
                ),
                width=500,
                position=10000,
                angle=0,
            ),
            width=500,
            position=10000,
            angle=0,
        ),
        width=500,
        position=10000,
        angle=0,
    ),
]


drifting_streams = sudden_drifts + gradual_drifts
