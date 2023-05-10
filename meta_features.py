import tsfel
from typing import List, Dict
import random
import pandas as pd

# Features
# abslute energy ok
# total energy ok
# centroid ok
# entropy ok
# Area under the curve
# Average number of elements by class
# Percentage of elements of the minority class
# Percentage of elements of the majority class
# Number of classes
# Number of attributes
# Number of numeric attributes
# Number of nominal (symbolic) attribute
# Fisher's discriminant ratio
# Overlapping the per-class bounding boxes
# Maximum individual feature efficiency

# attributes Number of attributes
# numeric Number of numerical attributes
# nominal Number of nominal attributes
# samples Number of examples
# dimension samples/attributes
# numRate numeric/attributes
# nomRate nominal/attributes
# symbols (min, max, mean, sd, sum) Distributions of categories in attributes
# classes (min, max, mean, sd) Classes distributions
# Statistical (ST)
# sks Skewness
# sksP Skewness for normalized dataset
# kts Kurtosis
# ktsP Kurtosis for normalized datasets
# absC Correlation between attributes
# canC Canonical correlation between matrices
# frac Fraction of canonical correlation
# Information-theoretic (IN)
# clEnt Class entropy
# nClEnt Class entropy for normalized dataset
# atrEnt Mean entropy of attributes
# nAtrEnt Mean entropy of attributes for
# normalized dataset
# jEnt Joint entropy
# mutInf Mutual information
# eqAtr clEnt/mutInf
# noiSig (atrEnt - mutInf)/MutIn


def extract_meta_features(features: List, classes: List) -> Dict:
    meta_features = {}
    cfg = tsfel.get_features_by_domain()
    tsfel.time_series_features_extractor
    meta_features["abs_energy"] = tsfel.abs_energy(features)
    # meta_features["total_energy"] = tsfel.total_energy(features)
    # meta_features["centroid"] = tsfel.calc_centroid(features)
    meta_features["entropy"] = tsfel.entropy(features)

    max_number_of_classes = pd.Series(classes).value_counts().max()
    min_number_of_classes = pd.Series(classes).value_counts().min()

    print(max_number_of_classes)
    print(min_number_of_classes)


if __name__ == "__main__":
    chunk = [
        (
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
            random.uniform(0, 1),
        )
        for i in range(0, 5)
    ]

    classes = [(random.choice([0, 1])) for i in range(0, 5)]

    print(classes)

    # print(chunk)

    extract_meta_features(chunk, classes)
