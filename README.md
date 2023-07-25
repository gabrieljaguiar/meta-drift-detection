# Enhancing Concept Drift Detection in Drifting and Imbalanced Data Streams through Meta-Learning


Learning from data streams is among the most important research topics in contemporary Machine Learning. One of the biggest challenges in this domain relies on proposing algorithms that can adapt to new arriving data. However, due to the evolving nature of data streams, they are subject to a phenomenon known as concept drift that makes previously learned knowledge outdated and must be efficiently detected in order to efficiently adapt the learning model. While there exists a plethora of drift detectors, with different mechanisms, selecting the most suitable for a new stream is a difficult task, since apriori knowledge may not be available and changes over time can affect the efficiency of the detector.  With this in mind, we propose a framework that exploits statistical and temporal meta-features from sliding windows to recommend a suitable drift detector for new unseen data using Meta-Learning. We performed experiments on 10 real-world data streams and 18 synthetic generated data streams that were subject to concept drift and class imbalance in order to evaluate the performance of the proposed framework. Experiments exposed that the proposed approach was able to enhance the concept drift detection in a variety of scenarios demonstrating robustness to class imbalance and highlighting the importance of dynamically select the drift detector.


*Manuscript submitted for publication at IEEE International Conference on Big Data 2023* (http://bigdataieee.org/BigData2023/)


## Implementation Details

This framework was implemented using Python 3.8 and the river library.
