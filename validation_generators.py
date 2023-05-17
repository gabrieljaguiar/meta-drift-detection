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
