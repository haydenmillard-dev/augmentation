from enum import Enum

class Strategy(Enum):
    NONE = None
    BASIC = "basic"
    ADVANCED = "advanced"
    MULTI = "multi"

class Mapper(Enum):
    NONE = None
    INPUT = "input"
    MULTI_LAYER = "multi-layer"
    DEEP_INPUT = "deep-input"
