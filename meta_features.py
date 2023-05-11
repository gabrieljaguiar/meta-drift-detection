import tsfel
from typing import List, Dict
import random
import pandas as pd
import numpy as np

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
# numerator <- function(j, data) {

#  tmp <- branch(data, j)
#  aux <- nrow(tmp) * (colMeans(tmp) -
#    colMeans(data[,-ncol(data), drop=FALSE]))^2
#  return(aux)
# }

# denominator <- function(j, data) {

#  tmp <- branch(data, j)
#  aux <- rowSums((t(tmp) - colMeans(tmp))^2)
#  return(aux)
# }

# call denominator and numerator for each class
# aux <- rowSums(do.call("cbind", num)) /
#    rowSums(do.call("cbind", den))

# Overlapping the per-class bounding boxes
# compute max and minimum value of each column
#   over <- colMax(rbind(colMin(maxmax) - colMax(minmin), 0))
#   rang <- colMax(maxmax) - colMin(minmin)
#   aux <- prod(over/rang, na.rm=TRUE)

# Maximum individual feature efficiency
#   data <- ovo(data)
#  aux <- mapply(function(d) {
#    colSums(nonOverlap(d))/nrow(d)
#  }, d=data)

# aux <- 1 - mean(colMax(aux))
#  aux <- 1 - colMax(aux)
#  return(aux)

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


def extract_meta_features(
    X: np, y: List, summary: List = None, tsfel_config: Dict = None
) -> Dict:
    if tsfel_config is None:
        domain = tsfel.get_features_by_domain()
        cfg = {}
        cfg["temporal"] = domain.get("temporal")

        print(cfg)
    else:
        cfg = tsfel_config

    if summary is None:
        summary = ["max", "min", "mean", "var"]

    fs = X.shape[0] * 0.2
    tsfel_features = tsfel.calc_window_features(cfg, X, fs=fs)
    summarized = tsfel_features.T.groupby(lambda x: x.split("_", 1)[1]).agg(summary).T
    flat = summarized.unstack().sort_index(level=1)
    flat.columns = flat.columns.map("_".join)

    pd.DataFrame(flat).to_csv("meta_features.csv")

    # print(flat.columns)


if __name__ == "__main__":
    chunk = [
        {
            "col1": random.uniform(0, 1),
            "col2": random.uniform(0, 1),
            "col3": random.uniform(0, 1),
            "col4": random.uniform(0, 1),
        }
        for i in range(0, 100)
    ]

    classes = [(random.choice([0, 1])) for i in range(0, 100)]

    df = pd.DataFrame(chunk)

    # print(chunk)

    extract_meta_features(df, classes)
