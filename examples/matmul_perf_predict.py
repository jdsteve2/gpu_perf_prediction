import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.statistics import get_op_poly, get_DRAM_access_poly, get_barrier_poly
import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel

# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

devices = ctx.get_info(cl.context_info.DEVICES)
print "devices: \n", devices

n = 2**12
a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
c_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)

order = "C"
knl = lp.make_kernel(
	"{[i,j,k]: 0<=i,j,k<%d}" % n,
	[
	    "c[i, j] = sum(k, a[i, k]*b[k, j])"
	    ],
	[
	    lp.GlobalArg("a", np.float32, shape=(n, n), order=order),
	    lp.GlobalArg("b", np.float32, shape=(n, n), order=order),
	    lp.GlobalArg("c", np.float32, shape=(n, n), order=order),
	    ],
	name="matmul")

ref_knl = knl  # TODO test result

knl = lp.split_iname(knl, "i", 16,
	outer_tag="g.0", inner_tag="l.1")
knl = lp.split_iname(knl, "j", 16,
	outer_tag="g.1", inner_tag="l.0")
knl = lp.split_iname(knl, "k", 16)
knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"])
knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner", ])

barrier_poly = get_barrier_poly(knl)
barrier_count = barrier_poly.eval_with_dict({'n': n})
print "barrier count: ", barrier_count

op_map = get_op_poly(knl)
flops = op_map.dict[np.dtype(np.float32)].eval_with_dict({'n': n})
print "flops: ", flops

sub_map = get_DRAM_access_poly(knl)  # noqa
print(sub_map)

f32coal_l = sub_map.dict[
                    (np.dtype(np.float32), 'consecutive', 'load')
                    ].eval_with_dict({'n': n})
f32coal_s = sub_map.dict[
                    (np.dtype(np.float32), 'consecutive', 'store')
                    ].eval_with_dict({'n': n})
f32coal = f32coal_l + f32coal_s
print "coalesced: %i, (stores: %i, loads: %i)" % (f32coal, f32coal_s, f32coal_l)
print "="*40

# execute
# -------
print("running kernel...")
knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev, c=c_mat_dev)
evt.wait()
print("actual runtime: ", (evt.profile.END - evt.profile.START)*1e-9)

gstats = GPUStats('TeslaK20')
kstats = KernelStats(flops/(n*n), 0, f32coal/(n*n), barrier_count)
tconfig = ThreadConfig(16*16, n/16*n/16)

model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32))
cycles = model.compute_exec_cycles()
print "total predicted execution cycles: ", cycles
print "total predicted time: ", cycles/(gstats.sm_clock_freq*10**9)

