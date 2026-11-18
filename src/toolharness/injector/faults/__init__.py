"""Fault type -> corruptor. Each of #9-#13 adds one import and one entry here."""
from ...data.scenario import FaultType
from ..corruptors import Corruptor, NoFault
from .hard_failure import HardFailure
from .malformed import Malformed

CORRUPTORS: dict[FaultType, Corruptor] = {
    FaultType.NONE: NoFault(),
    FaultType.HARD_FAILURE: HardFailure(),
    FaultType.MALFORMED: Malformed(),
}
