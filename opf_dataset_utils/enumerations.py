from enum import Enum, IntEnum


class SolutionBusIndices(IntEnum):
    VOLTAGE_ANGLE = 0
    VOLTAGE_MAGNITUDE = 1


class SolutionGeneratorIndices(IntEnum):
    ACTIVE_POWER = 0
    REACTIVE_POWER = 1


class SolutionACLineIndices(IntEnum):
    ACTIVE_POWER_TO = 0
    REACTIVE_POWER_TO = 1
    ACTIVE_POWER_FROM = 2
    REACTIVE_POWER_FROM = 3


class SolutionTransformerIndices(IntEnum):
    ACTIVE_POWER_TO = 0
    REACTIVE_POWER_TO = 1
    ACTIVE_POWER_FROM = 2
    REACTIVE_POWER_FROM = 3


class GridLoadIndices(IntEnum):
    ACTIVE_POWER = 0
    REACTIVE_POWER = 1


class GridShuntIndices(IntEnum):
    SUSCEPTANCE = 0
    CONDUCTANCE = 1


class GridACLineIndices(IntEnum):
    ANGLE_DIFF_MIN = 0
    ANGLE_DIFF_MAX = 1
    CHARGING_SUSCEPTANCE_FROM = 2
    CHARGING_SUSCEPTANCE_TO = 3
    SERIES_RESISTANCE = 4
    SERIES_REACTANCE = 5
    LONG_TERM_RATING = 6
    SHORT_TERM_RATING = 7
    EMERGENCY_RATING = 8


class GridTransformerIndices(IntEnum):
    ANGLE_DIFF_MIN = 0
    ANGLE_DIFF_MAX = 1
    SERIES_RESISTANCE = 2
    SERIES_REACTANCE = 3
    LONG_TERM_RATING = 4
    SHORT_TERM_RATING = 5
    EMERGENCY_RATING = 6
    TAP_MAGNITUDE = 7
    TAP_PHASE_SHIFT = 8
    CHARGING_SUSCEPTANCE_FROM = 9
    CHARGING_SUSCEPTANCE_TO = 10


class EdgeIndexIndices(IntEnum):
    FROM = 0
    TO = 1


class NodeTypes(str, Enum):
    BUS = "bus"
    GENERATOR = "generator"
    LOAD = "load"
    SHUNT = "shunt"


class EdgeTypes(str, Enum):
    AC_LINE = "ac_line"
    TRANSFORMER = "transformer"
    GENERATOR_LINK = "generator_link"
    LOAD_LINK = "load_link"
    SHUNT_LINK = "shunt_link"
