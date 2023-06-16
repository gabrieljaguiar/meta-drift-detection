import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

meta_data = pd.read_csv("merged.csv")
meta_data.drop(["idx", "delta_value", "stream", "chunk"], axis=1, inplace=True)

meta_data.loc[meta_data["drift_position"] > 0, "drift_position"] = 1
print(meta_data.columns)
meta_data = meta_data.fillna(0)

meta_model = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier())])

feature_columns = meta_data.columns.difference(["drift_position"])

print(meta_data["drift_position"].value_counts())
scores = cross_val_score(
    meta_model,
    meta_data.loc[:, feature_columns],
    meta_data.loc[:, "drift_position"],
    cv=2,
    n_jobs=1,
    scoring="balanced_accuracy"
)

print(scores)
