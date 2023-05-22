import pandas as pd


result_csv = pd.read_csv("results_FIXED.csv", index_col=0)


no_drift = result_csv[result_csv["drift_position"] == 0]
with_drift = result_csv[result_csv["drift_position"] > 0]


no_drift.to_csv("results_FIXED_no_drift.csv", index=None)
with_drift.to_csv("results_FIXED_with_drift.csv", index=None)
