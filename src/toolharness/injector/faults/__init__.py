"""Fault type -> corruptor.

CORRUPTORS holds the faults that need no extra data. build_corruptors adds the ones that apply a
pre-generated fact edit (silently_wrong, stale and contradiction).
"""
from ...data.fact_edits import FactEdit
from ...data.scenario import FaultType
from ..corruptors import Corruptor, NoFault
from .contradiction import Contradiction
from .hard_failure import HardFailure
from .malformed import Malformed
from .silently_wrong import SilentlyWrong
from .stale import Stale

CORRUPTORS: dict[FaultType, Corruptor] = {
    FaultType.NONE: NoFault(),
    FaultType.HARD_FAILURE: HardFailure(),
    FaultType.MALFORMED: Malformed(),
}


def build_corruptors(fact_edits: list[FactEdit]) -> dict[FaultType, Corruptor]:
    """Every corruptor, including the ones built from pre-generated data."""
    corruptors = dict(CORRUPTORS)
    corruptors[FaultType.SILENTLY_WRONG] = SilentlyWrong(fact_edits)
    corruptors[FaultType.STALE] = Stale(fact_edits)
    corruptors[FaultType.CONTRADICTION] = Contradiction(fact_edits)
    return corruptors
