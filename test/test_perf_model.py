import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel

gstats = GPUStats('HKexample')
kstats = KernelStats(27, 6, 0, 6)
tconfig = ThreadConfig(128, 80)

model = PerfModel(gstats, kstats, tconfig, 'float')
print model.compute_time()
