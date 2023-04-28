import random
from river import datasets


class BinaryImbalancedStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        generator: datasets.base.SyntheticDataset,
        imbalanceRatio: float,
        seed: int = 42,
    ):
        self.generator = generator
        self.imbalanceRatio = imbalanceRatio
        self.seed = seed
        self._rng = random.Random(self.seed)

        assert (self.generator.n_classes == 2, "Binary generators only")
        self.generatorIterator = iter(self.generator)

        super().__init__(
            self.generator.task,
            self.generator.n_features,
            self.generator.n_samples,
            self.generator.n_classes,
            self.generator.n_outputs,
        )

    def __iter__(self):
        expectedClass = 1 if self._rng.random() < self.imbalanceRatio else 0
        x, y = next(self.generatorIterator)
        while y != expectedClass:
            x, y = next(self.generatorIterator)
        yield x, y


class MultiClassImbalancedStream(datasets.base.SyntheticDataset):
    def __init__(
        self,
        generator: datasets.base.SyntheticDataset,
        imbalanceRatio: list,
        seed: int = 42,
    ):
        self.generator = generator
        self.imbalanceRatio = imbalanceRatio
        self.seed = seed
        self._rng = random.Random(self.seed)
        self.n_classes = self.generator.n_classes

        assert (sum(self.imbalanceRatio) == 1, "Sum of probabilities must be 1")
        assert (
            self.n_classes == len(self.imbalanceRatio),
            "Generator number of classes"
            + "and probability list should have the same size",
        )
        self.generatorIterator = iter(self.generator)

        super().__init__(
            self.generator.task,
            self.generator.n_features,
            self.generator.n_samples,
            self.generator.n_classes,
            self.generator.n_outputs,
        )

    def __iter__(self):
        nextClassProbability = self._rng.random()
        classIndex = -1

        while nextClassProbability > 0:
            classIndex += 1
            nextClassProbability -= self.imbalanceRatio[classIndex]
        expectedClass = classIndex
        x, y = next(self.generatorIterator)
        while y != expectedClass:
            x, y = next(self.generatorIterator)
        yield x, y
