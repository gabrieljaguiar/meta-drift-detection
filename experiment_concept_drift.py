# HERE IMPORT GENERATED VALIDATION DATA STREAMS
# TRAIN A MODEL ON META_FEATURES
# EVAL THEM USING HT AND ADWIN VALUES GIVEN BY THE MODEL
# EVAL THE PERFORMANCE OF CONCEPT DRIFT DETECTOR WITH MAJORITY VALUE (BASELINE)
from utils import concept_drift
from utils import evaluator
from river.datasets import synth
from river.drift import adwin, binary
from river.tree import HoeffdingTreeClassifier
from train_generators import drifiting_streams, META_STREAM_SIZE
import pandas as pd
from tqdm import tqdm
import os
from utils import adaptiveADWIN

model = HoeffdingTreeClassifier()

drift_detector = adaptiveADWIN.AdaptiveADWIN(delta=0.5)
