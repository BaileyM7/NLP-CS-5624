"""Fault type -> corruptor. Each of #9-#13 adds one import and one entry here."""
from ...data.scenario import FaultType
from ..corruptors import Corruptor, NoFault

CORRUPTORS: dict[FaultType, Corruptor] = {
    FaultType.NONE: NoFault(),
}
