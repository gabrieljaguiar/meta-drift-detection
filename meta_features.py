import tsfel
from typing import List, Dict
import random


# Features
# abslute energy
# total energy
# centroid
# entropy
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


def extract_meta_features(instance: List) -> Dict:
    cfg = tsfel.get_features_by_domain()
    tsfel.time_series_features_extractor
    meta_featuers = tsfel.mean_abs_diff(instance)
    print(meta_featuers)


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

    print(chunk)

    extract_meta_features(chunk)
