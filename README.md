# meta-drift-detection


Step by step


1. define generators with concept drift. (sudden, gradual and incremental)
2. define meta-features.
3. define the window which meta-features are going to be extracted.
4. define how the best adwin sensitivity value is going to be selected.
5. train a RF or SVM on meta-dataset.
6. start a stream with a classifier that depends on adwin and compare the performance with meta-adjusted with default parameters.
    6.1. predictive performance + performance of drift detection (number of instances before drift is detected)