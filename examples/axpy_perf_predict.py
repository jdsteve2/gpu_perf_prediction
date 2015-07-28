import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.statistics import get_op_poly, get_DRAM_access_poly, get_barrier_poly
import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
import islpy as isl

# setup
# -----
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

n = 2**24
x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)

dtype = np.float32
knl = lp.make_kernel(
    "[n] -> {[i]: 0<=i<%d}" % n,
    [
        "z[i] = 5.0*x[i]+7.0*y[i]"
        ],
    [
        lp.GlobalArg("x", dtype, shape=n),
        lp.GlobalArg("y", dtype, shape=n),
        lp.GlobalArg("z", dtype, shape=n),
        ],
    name="axpy")

ref_knl = knl 

BLOCKSIZE = 256
unroll = 4
knl = lp.split_iname(knl, "i", unroll*BLOCKSIZE,
     outer_tag="g.0", slabs=(0, 1))
knl = lp.split_iname(knl, "i_inner", BLOCKSIZE,
     outer_tag="unr", inner_tag="l.0")

check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=False)
#print "Correctness check: \n", check

# use ptx src to determine resource usage
cknl = lp.compiled.CompiledKernel(ctx, knl)
ptx_src = cknl.cl_kernel_info().cl_kernel.program.binaries[0]
ptx_src_file = open(knl.name+".ptx", 'w')
ptx_src_file.write(ptx_src)

barrier_poly = get_barrier_poly(knl)
barrier_count = barrier_poly.eval_with_dict({'n': n})
op_map = get_op_poly(knl)
flops = op_map.dict[np.dtype(np.float32)].eval_with_dict({'n': n})
sub_map = get_DRAM_access_poly(knl)  # noqa

f32coal_l = sub_map.dict.get(
                    (np.dtype(np.float32), 'consecutive', 'load')
                    ,isl.PwQPolynomial('{ 0 }')
                    ).eval_with_dict({'n': n})
f32coal_s = sub_map.dict.get(
                    (np.dtype(np.float32), 'consecutive', 'store')
                    ,isl.PwQPolynomial('{ 0 }')
                    ).eval_with_dict({'n': n})
f32coal = f32coal_l + f32coal_s
#print "coalesced: %i, (stores: %i, loads: %i)" % (f32coal, f32coal_s, f32coal_l)
f32uncoal_l = sub_map.dict.get(
                    (np.dtype(np.float32), 'nonconsecutive', 'load')
                    ,isl.PwQPolynomial('{ 0 }')
                    ).eval_with_dict({'n': n})
f32uncoal_s = sub_map.dict.get(
                    (np.dtype(np.float32), 'nonconsecutive', 'store')
                    ,isl.PwQPolynomial('{ 0 }')
                    ).eval_with_dict({'n': n})
f32uncoal = f32uncoal_l + f32uncoal_s
#print "uncoalesced: %i, (stores: %i, loads: %i)" % (f32uncoal, f32uncoal_s, f32uncoal_l)

print "="*40+"PTX SOURCE"
print "PTX source written to "+knl.name+".ptx"
print "To determine resource usage from PTX source, do:"
print "ptxas -v --gpu-name <compute capability> <filename.ptx>"
print "For example, with compute capability 3.5, do:"
print "ptxas -v --gpu-name sm_35 "+knl.name+".ptx"
print "="*40

print "="*40+"DEVICES"
print ctx.get_info(cl.context_info.DEVICES)
print "="*40

print "="*40+"KERNEL STATS"
print "barrier count: ", barrier_count
print "flops: ", flops
print(sub_map)
print "="*40

# execute
# -------
print "="*40+"TIMING RESULTS"
print "running kernel..."
#knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
evt, (out,) = knl(queue, x=x_vec_dev, y=y_vec_dev, z=z_vec_dev)
evt.wait()

gstats = GPUStats('TeslaK20')
total_threads = n/unroll
kstats = KernelStats(float(flops)/total_threads, float(f32uncoal)/total_threads, 
                     float(f32coal)/total_threads, float(barrier_count))
tconfig = ThreadConfig(BLOCKSIZE, n/(BLOCKSIZE*unroll))

model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32))
cycles = model.compute_exec_cycles()
print "actual runtime: ", (evt.profile.END - evt.profile.START)*1e-9
print "total predicted time: ", cycles/(gstats.sm_clock_freq*10**9)
print "total predicted execution cycles: ", cycles
print "="*40


