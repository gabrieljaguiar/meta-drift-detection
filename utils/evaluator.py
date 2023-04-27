import random
import math
import river
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

    def addResult(self, instance: Dict[float, int], classVotes: list):
        _, y = instance
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
    windowsize = 4
    evaluator = Evaluator(windowSize=windowsize)

    instances = [
        (1.0, 1),
        (1.0, 0),
        (1.0, 1),
        (1.0, 1),
        (1.0, 0),
        (1.0, 0),
        (1.0, 1),
        (1.0, 0),
        (1.0, 1),
        (1.0, 1),
        (1.0, 0),
        (1.0, 0),
        (1.0, 1),
        (1.0, 1),
        (1.0, 0),
        (1.0, 1),
        (1.0, 1),
        (1.0, 0),
        (1.0, 0),
        (1.0, 1),
    ]

    predictions = [
        (0.2, 0.8),  # 0
        (0.9, 0.1),  # 1
        (0.2, 0.8),  # 0
        (0.4, 0.6),  # 1
        (0.2, 0.8),  # 1
        (0.3, 0.7),  # 1
        (0.6, 0.4),  # 0
        (0.9, 0.1),  # 0
        (0.6, 0.4),  # 0
        (0.2, 0.8),  # 1
        (0.1, 0.9),  # 1
        (0.6, 0.4),  # 0
        (0.3, 0.7),  # 1
        (0.4, 0.6),  # 1
        (0.9, 0.1),  # 0
        (0.8, 0.2),  # 0
        (0.7, 0.3),  # 1
        (0.4, 0.6),  # 1
        (0.2, 0.8),  # 1
        (0.6, 0.4),  # 0
    ]

    for i, instance, prediction in zip(range(len(instances)), instances, predictions):
        evaluator.addResult(instance, prediction)
        if (i + 1) % windowsize == 0:
            print(evaluator.getAccuracy())
