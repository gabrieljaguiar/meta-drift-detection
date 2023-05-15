from river import datasets
from river.datasets import synth
from typing import Dict
import random
import math


class IncrementalConceptDrift(datasets.base.SyntheticDataset):
    def __init__(
        self,
        initialStream: datasets.base.SyntheticDataset,
        nextStream: datasets.base.SyntheticDataset,
        width: int,
        startingPosition: int,
    ):
        self.initialStream = initialStream
        self.nextStream = nextStream
        self.width = width
        self.startingPosition = startingPosition
        self.instanceCount = 0
        super().__init__(
            self.initialStream.task,
            self.initialStream.n_features,
            self.initialStream.n_samples,
            self.initialStream.n_classes,
            self.initialStream.n_outputs,
        )

        self.initialStreamIterator = iter(self.initialStream)
        self.nextStreamIterator = iter(self.nextStream)

    def __interpolate(self, instance1: Dict, instance2: Dict, driftProbabilty: float):
        # Only interpolates numbers, add ifs to interpolate nominal attributes
        for key in instance1.keys():
            value1 = instance1.get(key)
            value2 = instance2.get(key)
            interpolate = (1 - driftProbabilty) * value1 + driftProbabilty * value2
            instance1[key] = interpolate
        return instance1

    def __iter__(self):
        while True:
            self.instanceCount += 1
            x = -4.0 * (self.instanceCount - self.position) / self.width
            try:
                driftProbability = 1.0 / (1.0 + math.exp(x))
            except:
                driftProbability = 0

            try:
                primaryStreamNextElement = next(self.initialStreamIterator)
                driftStreamNextElement = next(self.initialStreamIterator)
                while driftStreamNextElement[1] != primaryStreamNextElement[1]:
                    driftStreamNextElement = next(self.initialStreamIterator)

                x, y = primaryStreamNextElement
                newX = self.__interpolate(x, driftStreamNextElement[0])
                nextElement = (newX, y)

            except StopIteration:
                break
            yield nextElement


class ConceptDriftStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        initialStream: datasets.base.SyntheticDataset,
        nextStream: datasets.base.SyntheticDataset,
        width: int,
        position: int,
        size: int,
        angle: float = 0,
    ):
        self.initialStream = initialStream
        self.nextStream = nextStream
        self.width = width
        self.position = position
        self.angle = angle
        self.instanceCount = 0
        self.size = size
        self.name = initialStream.__class__
        super().__init__(
            self.initialStream.task,
            self.initialStream.n_features,
            self.initialStream.n_samples,
            self.initialStream.n_classes,
            self.initialStream.n_outputs,
        )

        self.initialStreamIterator = iter(self.initialStream)
        self.nextStreamIterator = iter(self.nextStream)

    def __iter__(self):
        while True:
            self.instanceCount += 1
            if self.instanceCount == self.size:
                break

            x = -4.0 * (self.instanceCount - self.position) / self.width
            try:
                driftProbability = 1.0 / (1.0 + math.exp(x))
            except:
                driftProbability = 0

            try:
                nextElement = (
                    next(self.initialStreamIterator)
                    if random.random() > driftProbability
                    else next(self.nextStreamIterator)
                )
            except StopIteration:
                break
            yield nextElement

    def reset(self):
        self.instanceCount = 0
        self.initialStreamIterator = iter(self.initialStream)
        self.nextStreamIterator = iter(self.nextStream)


if __name__ == "__main__":
    stream1 = synth.Agrawal(classification_function=0, seed=42)
    stream2 = synth.Agrawal(classification_function=8, seed=42)

    conceptDriftStream = ConceptDriftStream(
        stream1, stream2, width=1, position=500, angle=0
    )

    slicer = conceptDriftStream.take(506)

    for x, y in slicer:
        print(x, y)
