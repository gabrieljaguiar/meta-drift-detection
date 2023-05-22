from river.drift import ADWIN
from utils.adaptiveWindow import AdaptiveWindowing


class AdaptiveADWIN(ADWIN):
    def _reset(self):
        super()._reset()
        self._helper = AdaptiveWindowing(
            delta=self.delta,
            clock=self.clock,
            max_buckets=self.max_buckets,
            min_window_length=self.min_window_length,
            grace_period=self.grace_period,
        )

    def updateDelta(self, delta: float):
        self._helper.delta = delta
