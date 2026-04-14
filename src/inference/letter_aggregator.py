"""Агрегация P(target) по буквам внутри trial."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrialAggregator:
    """Копит P(target) по буквам, возвращает предсказание.

    Parameters
    ----------
    target_letter : str | None
        Истинный таргет (для лога accuracy; обычно известен в ground-truth
        спеллере).
    average_last : int | None
        Усреднять только последние N репетиций на букву.
        None → усреднять все. (В задаче: репетиций 10, усреднять по 5 —
        ставим 5, если хочется отключить шумные ранние.)

    """

    target_letter: str | None = None
    average_last: int | None = None
    scores: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list),
    )

    def add(self, letter: str, p_target: float) -> None:
        self.scores[letter].append(float(p_target))

    def counts(self) -> dict[str, int]:
        return {l: len(s) for l, s in self.scores.items()}

    def min_count(self) -> int:
        if not self.scores:
            return 0
        return min(len(s) for s in self.scores.values())

    def summary(self) -> dict[str, float]:
        """{буква: средний P(target) по последним average_last эпохам}."""
        out = {}
        for letter, lst in self.scores.items():
            arr = np.asarray(lst, dtype=np.float64)
            if self.average_last is not None and arr.size > self.average_last:
                arr = arr[-self.average_last:]
            if arr.size == 0:
                continue
            out[letter] = float(arr.mean())
        return out

    def predict(self) -> tuple[str | None, dict[str, float]]:
        s = self.summary()
        if not s:
            return None, {}
        best = max(s, key=s.get)
        return best, s

    def reset(self) -> None:
        self.scores = defaultdict(list)
        self.target_letter = None
