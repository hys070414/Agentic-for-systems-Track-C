from .precision_strategy import PrecisionStrategy, PrecisionProfile, HardwareInfo
from .decomposition_strategy import DecompositionStrategy, KernelSpecRef, KernelTuningParams

strategy = DecompositionStrategy()
precision_strategy = PrecisionStrategy()
hardware = HardwareInfo()
