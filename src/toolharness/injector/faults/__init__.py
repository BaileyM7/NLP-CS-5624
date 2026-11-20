"""Fault type -> corruptor.

CORRUPTORS holds the faults that need no extra data. build_corruptors adds the ones that need
pre-generated data (silently_wrong needs fact edits).
"""
from ...data.fact_edits import FactEdit
from ...data.scenario import FaultType
from ..corruptors import Corruptor, NoFault
from .hard_failure import HardFailure
from .malformed import Malformed
from .silently_wrong import SilentlyWrong

CORRUPTORS: dict[FaultType, Corruptor] = {
    FaultType.NONE: NoFault(),
    FaultType.HARD_FAILURE: HardFailure(),
    FaultType.MALFORMED: Malformed(),
}


def build_corruptors(fact_edits: list[FactEdit]) -> dict[FaultType, Corruptor]:
    """Every corruptor, including the ones built from pre-generated data."""
    corruptors = dict(CORRUPTORS)
    corruptors[FaultType.SILENTLY_WRONG] = SilentlyWrong(fact_edits)
    return corruptors
