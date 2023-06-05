python meta_features.py --output training_set_1_added_features.csv --feature-set 1 --n-jobs 32

python meta_features.py --output training_set_1_added_features.csv --feature-set 1 --n-jobs 1


python experiment_concept_drift.py --model META --mf training_set_1_added_features.csv --output results_META_1_1500_500.csv --mt 1500 --st 500 --n-jobs 32
python experiment_concept_drift.py --model META --mf training_set_1_added_features.csv --output results_META_1_10000_1000.csv --mt 10000 --st 10000 --n-jobs 32