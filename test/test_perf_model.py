import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
import numpy as np

gstats = GPUStats('HKexample')
kstats = KernelStats(27, 6, 0, 6)
tconfig = ThreadConfig(128, 80)

model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32), active_blocks=5)
print("total predicted execution cycles: ",  model.compute_exec_cycles())
