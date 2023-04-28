import random
import math
import river
from river.datasets import synth
from typing import Dict


class Evaluator:
    def __init__(self, windowSize, numberOfClasses=2):
        self.windowSize = windowSize
        self.totalObservedInstances = 0
        self.predictions = [0] * self.windowSize
        self.numberOfClasses = numberOfClasses
        self.rowKappa = [0.0] * self.numberOfClasses
        self.columnKappa = [0.0] * self.numberOfClasses
        self.cm = [[0.0] * self.numberOfClasses] * self.numberOfClasses

    def addResult(self, instance: Dict[float, int], probabilties: Dict):
        _, y = instance
        classVotes = [probabilties.get(i, 0) for i in range(self.numberOfClasses)]
        prediction = classVotes.index(max(classVotes))
        self.predictions[self.totalObservedInstances % self.windowSize] = (
            1 if prediction == y else 0
        )
        self.totalObservedInstances += 1

    def getAccuracy(self) -> float:
        accuracy = (
            sum(self.predictions[: self.totalObservedInstances])
            / self.totalObservedInstances
            if self.totalObservedInstances < self.windowSize
            else sum(self.predictions) / self.windowSize
        )

        return accuracy


if __name__ == "__main__":
    from river.tree import HoeffdingTreeClassifier
    import concept_drift

    windowsize = 500
    evaluator = Evaluator(windowSize=windowsize)

    stream1 = synth.Agrawal(classification_function=0, seed=42)
    stream2 = synth.Agrawal(classification_function=8, seed=42)

    conceptDriftStream = concept_drift.ConceptDriftStream(
        stream1, stream2, width=1, position=4000, angle=0
    )

    model = HoeffdingTreeClassifier(
        grace_period=100, delta=1e-5, nominal_attributes=["elevel", "car", "zipcode"]
    )

    idx = 0

    for x, y in conceptDriftStream.take(200):
        model.learn_one(x, y)

    for x, y in conceptDriftStream.take(6000):
        idx += 1
        y_hat = model.predict_proba_one(x)
        evaluator.addResult((x, y), y_hat)
        model.learn_one(x, y)
        if idx % windowsize == 0:
            print("Accuracy {}: {}%".format(idx, evaluator.getAccuracy()))
