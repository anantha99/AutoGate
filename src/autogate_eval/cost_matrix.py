"""The PRD cost matrix behind safety-weighted error (SWE).

Rows are the true route and columns the predicted route. A true LOCAL is
split in two: sending a safety-critical command (defrost, wipers, hazards)
anywhere but the head unit costs far more than doing so for a comfort
command or a local query. The matrix is data in ``configs/cost_matrix.json``
and is validated on load: every cell present, costs non-negative, and a
correct prediction free.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from autogate_bench.schema import Route

DEFAULT_COST_MATRIX_PATH = Path(__file__).resolve().parents[2] / "configs" / "cost_matrix.json"

LOCAL_SAFETY_CRITICAL = "LOCAL_SAFETY_CRITICAL"
LOCAL_OTHER = "LOCAL_OTHER"
TRUE_ROWS: tuple[str, ...] = (
    LOCAL_SAFETY_CRITICAL,
    LOCAL_OTHER,
    *(str(r) for r in Route if r is not Route.LOCAL),
)


def true_row(true_route: Route | str, safety_critical: bool) -> str:
    """The matrix row for a true route; only LOCAL is split by safety_critical."""
    true_route = Route(true_route)
    if true_route is Route.LOCAL:
        return LOCAL_SAFETY_CRITICAL if safety_critical else LOCAL_OTHER
    return str(true_route)


def _diagonal(row: str) -> Route:
    return Route.LOCAL if row in (LOCAL_SAFETY_CRITICAL, LOCAL_OTHER) else Route(row)


@dataclass(frozen=True)
class CostMatrix:
    costs: Mapping[str, Mapping[Route, float]]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostMatrix:
        raw = data.get("costs")
        if not isinstance(raw, Mapping):
            raise ValueError("cost matrix needs a 'costs' mapping")
        if set(raw) != set(TRUE_ROWS):
            missing, extra = sorted(set(TRUE_ROWS) - set(raw)), sorted(set(raw) - set(TRUE_ROWS))
            raise ValueError(
                f"cost matrix rows must be exactly {list(TRUE_ROWS)}; "
                f"missing {missing}, extra {extra}"
            )
        routes = {str(r) for r in Route}
        costs: dict[str, dict[Route, float]] = {}
        for row in TRUE_ROWS:
            cells = raw[row]
            if not isinstance(cells, Mapping) or set(cells) != routes:
                raise ValueError(f"cost matrix row {row} must have exactly the columns {routes}")
            parsed: dict[Route, float] = {}
            for col, v in cells.items():
                if isinstance(v, bool) or not isinstance(v, int | float) or v < 0:
                    raise ValueError(f"cost[{row}][{col}] must be a non-negative number")
                parsed[Route(col)] = float(v)
            if parsed[_diagonal(row)] != 0:
                raise ValueError(f"cost[{row}][{_diagonal(row)}] (a correct prediction) must be 0")
            costs[row] = parsed
        return cls(costs)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_COST_MATRIX_PATH) -> CostMatrix:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def cost(
        self, true_route: Route | str, pred_route: Route | str, safety_critical: bool
    ) -> float:
        return self.costs[true_row(true_route, safety_critical)][Route(pred_route)]

    def matrix(self, safety_critical: bool) -> list[list[float]]:
        """5x5 costs in ``Route`` order, with the LOCAL row chosen by ``safety_critical``."""
        return [[self.cost(t, p, safety_critical) for p in Route] for t in Route]


@lru_cache(maxsize=1)
def default_cost_matrix() -> CostMatrix:
    return CostMatrix.load()


def cost(true_route: Route | str, pred_route: Route | str, safety_critical: bool) -> float:
    """Cost of predicting ``pred_route`` when the label is ``true_route`` (default matrix)."""
    return default_cost_matrix().cost(true_route, pred_route, safety_critical)
