from river import datasets
from river.datasets import synth
import random
import math


class ConceptDriftStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        initialStream: datasets.base.SyntheticDataset,
        nextStream: datasets.base.SyntheticDataset,
        width: int,
        position: int,
        angle: float,
        stream_size: int = 100000,
    ):
        self.initialStream = initialStream
        self.nextStream = nextStream
        self.width = width
        self.position = position
        self.angle = angle
        self.instanceCount = 0
        self.stream_size = stream_size
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


if __name__ == "__main__":
    stream1 = synth.Agrawal(classification_function=0, seed=42)
    stream2 = synth.Agrawal(classification_function=8, seed=42)
    # print(stream1)

    conceptDriftStream = ConceptDriftStream(
        stream1, stream2, width=1, position=500, angle=0
    )

    slicer = conceptDriftStream.take(506)

    for x, y in slicer:
        print(x, y)
    # print(x)

    # stream_iter = iter(stream1)

    # print(next(stream_iter))
    # print(next(stream_iter))
    # print(next(stream_iter))
