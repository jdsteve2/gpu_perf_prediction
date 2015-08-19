from __future__ import division, print_function

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom  # noqa
from loopy.statistics import get_op_poly, get_DRAM_access_poly, get_barrier_poly
import sys
sys.path.append("../performance_model")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
import islpy as isl
import math

run_mm = True
run_axpy = True
run_tp = True

def main():
    # setup
    # -----
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
    lstsq_A = []
    lstsq_y = []
    predicted_times_HK = []
    actual_times_all = []

    '''
    print "="*40+"DEVICES"
    print ctx.get_info(cl.context_info.DEVICES)
    print "="*40
    '''
    trials_n = 4
    if run_mm:
        nvals = [2**(9+x) for x in range(trials_n)]
        configs_t = [(16, 8), (16, 16), (32, 16)]
        A_mm, y_mm, predicted_mm, actual_mm = run_mm_trials(ctx, queue, nvals,
                                                            configs_t, "allcoal")
        for row in range(len(A_mm)):
            lstsq_A.append(A_mm[row])
            lstsq_y.append(y_mm[row])
            predicted_times_HK.append(predicted_mm[row])
            actual_times_all.append(actual_mm[row])

        A_mm, y_mm, predicted_mm, actual_mm = run_mm_trials(ctx, queue, nvals,
                                                            configs_t, "partcoal")
        for row in range(len(A_mm)):
            lstsq_A.append(A_mm[row])
            lstsq_y.append(y_mm[row])
            predicted_times_HK.append(predicted_mm[row])
            actual_times_all.append(actual_mm[row])

    # now train on axpy
    if run_axpy:
        nvals = [2**(24+x) for x in range(trials_n)]
        #configs_t = [(16, 1), (32, 1), (64, 1), (128, 1), (256, 1), (512, 1)]
        configs_t = [(64, 1), (128, 1), (256, 1), (512, 1)]
        A_axpy, y_axpy, predicted_axpy, actual_axpy = run_axpy_trials(ctx, queue, nvals, configs_t)
        for row in range(len(A_axpy)):
            lstsq_A.append(A_axpy[row])
            lstsq_y.append(y_axpy[row])
            predicted_times_HK.append(predicted_axpy[row])
            actual_times_all.append(actual_axpy[row])

    if run_tp:
        nvals = [2**(10+x) for x in range(trials_n)]
        configs_t = [(8, 8), (16, 16), (24, 24), (32, 32)]
        A_tp, y_tp, predicted_tp, actual_tp = run_tp_trials(ctx, queue, nvals, configs_t)
        for row in range(len(A_tp)):
            lstsq_A.append(A_tp[row])
            lstsq_y.append(y_tp[row])
            predicted_times_HK.append(predicted_tp[row])
            actual_times_all.append(actual_tp[row])

    # least squares calculations
    if run_mm or run_axpy or run_tp :

        Atrain, ytrain, Atest, ytest = split_for_train_test(lstsq_A, lstsq_y)

        '''
        lstsq_weights, resid, q, q = np.linalg.lstsq(lstsq_A, lstsq_y)
        U, s, V = np.linalg.svd(lstsq_A, full_matrices=False)
        '''
        lstsq_weights, resid, q, q = np.linalg.lstsq(Atrain, ytrain)
        U, s, V = np.linalg.svd(Atrain, full_matrices=False)
        #  TODO y and actual are same, redundant
        cos_angles = []
        for row1 in range(len(ytrain)):
            for j in range(len(ytrain)-1-row1):
                row2 = row1+1+j
                cos_angles.append(cos_angle_btw(Atrain[row1], Atrain[row2]))
                #print(row1, row2, cos_angles[-1])

        print("Least Squares Residual:\n", np.dot(Atrain, lstsq_weights)-ytrain)
        print("Least Squares singular values:\n", s)
        print("="*40+"TIMING RESULTS")

        '''
        print("i\tactual\t\tpredicted\terror\t\tlstsq\t\terror")

        rel_error = []
        rel_error_lstsq = []
        for i in range(len(actual_times_all)):
            predicted = predicted_times_HK[i]
            predicted_lstsq = np.dot(lstsq_A[i], lstsq_weights)
            actual = actual_times_all[i]
            rel_error.append((predicted-actual)/actual)
            rel_error_lstsq.append((predicted_lstsq-actual)/actual)
            print("%i\t%f\t%f\t%f\t%f\t%f" % (i, actual, predicted, rel_error[i],
                                              predicted_lstsq, rel_error_lstsq[i]))
        '''
        print("i\tactual\t\tlstsq\t\terror")
        rel_error_lstsq = []
        for i in range(len(ytrain)):
            predicted_lstsq = np.dot(Atest[i], lstsq_weights)
            actual = ytest[i]
            rel_error_lstsq.append((predicted_lstsq-actual)/actual)
            print("%i\t%.7f\t%.7f\t%.7f" % (i, actual, predicted_lstsq, rel_error_lstsq[i]))

        print("i\tactual\t\tHK\t\terror")
        rel_error_HK = []
        for i in range(len(predicted_times_HK)):
            predicted = predicted_times_HK[i]
            actual = actual_times_all[i]
            rel_error_HK.append((predicted-actual)/actual)
            print("%i\t%.7f\t%.7f\t%.7f" % (i, actual, predicted, rel_error_HK[i]))
        print("avg relative error HK: ", np.average(np.absolute(rel_error_HK))) 
        print("avg relative error LS: ", np.average(np.absolute(rel_error_lstsq)))

        print(np.average(cos_angles))

        print_Ay(lstsq_A, lstsq_y)

        print(lstsq_weights)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def cos_angle_btw(v1, v2):
    """ Returns the cosine of angle between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.dot(v1_u, v2_u)

def print_ptx_src_msg(knl_name):
    print("="*40+"PTX SOURCE")
    print("PTX source written to "+knl_name+".ptx")
    print("To determine resource usage from PTX source, do:")
    print("ptxas -v --gpu-name <compute capability> <filename.ptx>")
    print("For example, with compute capability 3.5, do:")
    print("ptxas -v --gpu-name sm_35 "+knl_name+".ptx")
    print("="*40)

def print_Ay(A, y):
    for row in range(len(A)):
        print("%d\t" % (row), end='')
        for col in range(len(A[0])):
            #if abs(A[row][col]) > 10**5:
            if col < 2:
                print("%e\t" % (A[row][col]), end='')
            else:
                print("%e\t" % (A[row][col]), end='')
        print("| %f" % (y[row]))

def get_DRAM_f32_accesses(sub_map, param_dict):
    f32coal_l = sub_map.dict.get(
                        (np.dtype(np.float32), 'consecutive', 'load'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    f32coal_s = sub_map.dict.get(
                        (np.dtype(np.float32), 'consecutive', 'store'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    f32uncoal_l = sub_map.dict.get(
                        (np.dtype(np.float32), 'nonconsecutive', 'load'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    f32uncoal_s = sub_map.dict.get(
                        (np.dtype(np.float32), 'nonconsecutive', 'store'),
                        isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    return (f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s)

def get_32b_ops(op_map, param_dict):
    flops = op_map.dict.get(
                        np.dtype(np.float32),isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    iops = op_map.dict.get(
                        np.dtype(np.int32),isl.PwQPolynomial('{ 0 }')
                        ).eval_with_dict(param_dict)
    return (flops, iops)

def update_LS_matrix(A, flops, f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s,
                     barrier_ct, blocks, thread_work_units, itemsize, model):

    reps_per_SM = math.ceil(blocks/(model.active_blocks_per_SM*
                                    model.GPU_stats.SM_count))
    multiplier = reps_per_SM
    #multiplier = math.ceil(blocks/model.GPU_stats.SM_count)
    #multiplier = blocks
    # TODO assumes there are enough blocks to fully load all SMs
    A.append([multiplier*itemsize*flops/thread_work_units,
              multiplier*itemsize*f32uncoal_l/thread_work_units,
              multiplier*itemsize*f32coal_l/thread_work_units,
              multiplier*itemsize*f32uncoal_s/thread_work_units,
              multiplier*itemsize*f32coal_s/thread_work_units,
              multiplier*itemsize*abs(f32uncoal_s-f32uncoal_l)/thread_work_units,
              multiplier*itemsize*abs(f32coal_s-f32coal_l)/thread_work_units,
              multiplier*barrier_ct])

def split_for_train_test(A, y):

    Atrain = []
    Atest = []
    ytrain = []
    ytest = []

    for row in range(len(A)):
        #'''
        if row % 2 == 0:
            Atrain.append(A[row])
            ytrain.append(y[row])
        else:
            Atest.append(A[row])
            ytest.append(y[row])
        '''
        Atrain.append(A[row])
        ytrain.append(y[row])
        Atest.append(A[row])
        ytest.append(y[row])
        '''
    return (Atrain, ytrain, Atest, ytest)

def run_mm_trials(ctx, queue, nvals, configs_t, version):

    A = []
    y = []
    predicted = []
    actual = []

    #TODO figure out smem usage issue
    for n in nvals:
        a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
        b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
        c_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)

        order = "C"
        knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ],[
                lp.GlobalArg("a", np.float32, shape=(n, n), order=order),
                lp.GlobalArg("b", np.float32, shape=(n, n), order=order),
                lp.GlobalArg("c", np.float32, shape=(n, n), order=order),
            ], name="matmul")

        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            if version == "allcoal":
                knl = lp.split_iname(knl, "i", BSIZEy, outer_tag="g.0", inner_tag="l.1")
                knl = lp.split_iname(knl, "j", BSIZEx, outer_tag="g.1", inner_tag="l.0")
            elif version == "partcoal":
                knl = lp.split_iname(knl, "i", BSIZEy, outer_tag="g.0", inner_tag="l.0")
                knl = lp.split_iname(knl, "j", BSIZEx, outer_tag="g.1", inner_tag="l.1")
            else:
                1/0
                # TODO error
            knl = lp.split_iname(knl, "k", BSIZEy)
            knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"])
            knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner", ])

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True)
            #print "Correctness check: \n", check
            # use ptx src to determine resource usage
            """
            cknl = lp.compiled.CompiledKernel(ctx, knl)
            ptx_src = cknl.cl_kernel_info().cl_kernel.program.binaries[0]
            ptx_src_file = open(knl.name+".ptx", 'w')
            ptx_src_file.write(ptx_src)
            """

            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict({'n': n})

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, {'n': n})

            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(sub_map, {'n': n})
            f32coal = f32coal_l + f32coal_s
            f32uncoal = f32uncoal_l + f32uncoal_s

            '''
            print_ptx_src_msg(knl.name)
            print "="*40+"KERNEL STATS"
            print "barrier count: ", barrier_ct
            print "flops: ", flops
            print(sub_map)
            print "="*40
            '''

            # execute
            # -------
            #print "="*40+"TIMING RESULTS"
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
            evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev, c=c_mat_dev)
            evt.wait()

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 25
            shared_mem_per_block = 2*4*BSIZEx*BSIZEy
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_ct, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(np.float32))
            cycles = model.compute_total_cycles()

            actual.append((evt.profile.END - evt.profile.START)*1e-9)
            predicted.append(cycles/(gstats.sm_clock_freq*10**9))

            '''
            print "actual runtime: ", actual[-1]
            print "total predicted time: ", predicted[-1]
            print "total predicted execution cycles: ", cycles
            print "="*40
            '''
            update_LS_matrix(A, flops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(np.float32).itemsize, model)
            y.append(actual[-1])

    return (A, y, predicted, actual)

def run_axpy_trials(ctx, queue, nvals, configs_t):

    A = []
    y = []
    predicted = []
    actual = []

    #TODO figure out smem usage issue
    for n in nvals:
        x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
        y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
        z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)

        dtype = np.float32
        knl = lp.make_kernel(
            "[n] -> {[i]: 0<=i<%d}" % n,
            [
                "z[i] = 5.0*x[i]+7.0*y[i]"
            ],[
                lp.GlobalArg("x", dtype, shape=n),
                lp.GlobalArg("y", dtype, shape=n),
                lp.GlobalArg("z", dtype, shape=n),
            ], name="axpy")

        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            unroll = 4
            knl = lp.split_iname(knl, "i", unroll*BSIZEx,
                 outer_tag="g.0", slabs=(0, 1))
            knl = lp.split_iname(knl, "i_inner", BSIZEx,
                 outer_tag="unr", inner_tag="l.0")

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=False)
            #print "Correctness check: \n", check

            # use ptx src to determine resource usage
            '''
            cknl = lp.compiled.CompiledKernel(ctx, knl)
            ptx_src = cknl.cl_kernel_info().cl_kernel.program.binaries[0]
            ptx_src_file = open(knl.name+".ptx", 'w')
            ptx_src_file.write(ptx_src)
            '''
            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict({'n': n})
            op_map = get_op_poly(knl)
            op_map = get_op_poly(knl)

            flops, iops = get_32b_ops(op_map, {'n': n})
            sub_map = get_DRAM_access_poly(knl)  # noqa

            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(sub_map, {'n': n})
            f32coal = f32coal_l + f32coal_s
            f32uncoal = f32uncoal_l + f32uncoal_s

            '''
            print_ptx_src_msg(knl.name)

            print "="*40+"KERNEL STATS"
            print "barrier count: ", barrier_ct
            print "flops: ", flops
            print(sub_map)
            print "="*40
            '''

            # execute
            # -------
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
            evt, (out,) = knl(queue, x=x_vec_dev, y=y_vec_dev, z=z_vec_dev)
            evt.wait()

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 20
            shared_mem_per_block = 0
            total_blocks = math.ceil(n/(BSIZEx*unroll))
            kstats = KernelStats(flops*unroll/n, f32uncoal*unroll/n,
                                 f32coal*unroll/n, barrier_ct, reg32_per_thread,
                                 shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig, np.dtype(np.float32))
            cycles = model.compute_total_cycles()

            actual.append((evt.profile.END - evt.profile.START)*1e-9)
            predicted.append(cycles/(gstats.sm_clock_freq*10**9))

            update_LS_matrix(A, flops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n/unroll,
                             np.dtype(np.float32).itemsize, model)
            y.append(actual[-1])

    return (A, y, predicted, actual)

def run_tp_trials(ctx, queue, nvals, configs_t):

    A = []
    y = []
    predicted = []
    actual = []

    for n in nvals:
        a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
        b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
        order = "C"
        dtype = np.float32
        knl = lp.make_kernel(
                "{[i,j]: 0<=i,j<%d}" % n,
                [
                    "b[i, j] = a[j, i]"
                ],[
                    lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                    lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                ],
                name="transpose")

        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            knl = lp.split_iname(knl, "i", BSIZEy, outer_tag="g.0", inner_tag="l.1")
            knl = lp.split_iname(knl, "j", BSIZEx, outer_tag="g.1", inner_tag="l.0")
            knl = lp.add_prefetch(knl, 'a', ["i_inner", "j_inner"])

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True)
            #print "Correctness check: \n", check

            # use ptx src to determine resource usage
            '''
            cknl = lp.compiled.CompiledKernel(ctx, knl)
            ptx_src = cknl.cl_kernel_info().cl_kernel.program.binaries[0]
            ptx_src_file = open(knl.name+".ptx", 'w')
            ptx_src_file.write(ptx_src)
            '''

            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict({'n': n})

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, {'n': n})

            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(sub_map, {'n': n})
            f32coal = f32coal_l + f32coal_s
            f32uncoal = f32uncoal_l + f32uncoal_s

            # execute
            # -------
            #print "="*40+"TIMING RESULTS"
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
            evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev)
            evt.wait()

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 8
            shared_mem_per_block = 4*BSIZEx*BSIZEy
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_ct, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(np.float32))
            cycles = model.compute_total_cycles()

            actual.append((evt.profile.END - evt.profile.START)*1e-9)
            predicted.append(cycles/(gstats.sm_clock_freq*10**9))

            update_LS_matrix(A, flops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(np.float32).itemsize, model)
            y.append(actual[-1])

    return (A, y, predicted, actual)


"""
print("="*40+"TIMING RESULTS")
print("n\tBx\tBy\tactual\t\tpredicted\terror\t\tlstsq\t\terror")
rel_error = []
rel_error_lstsq = []
for i in range(len(actual_times_all)):
    predicted = predicted_times_HK[i]
    predicted_lstsq = np.dot(lstsq_A[i], lstsq_weights)
    actual = actual_times_all[i]
    rel_error.append((predicted-actual)/actual)
    rel_error_lstsq.append((predicted_lstsq-actual)/actual)
    print("%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f" %
                        (nvals[int(i/len(configs_t))], configs_t[i%len(configs_t)][0], 
                         configs_t[i%len(configs_t)][1], actual, predicted,
                         rel_error[i], predicted_lstsq, rel_error_lstsq[i]))
"""

"""
print("="*40+"TIMING RESULTS")
print("n\tBx\tBy\tactual\t\tpredicted\terror\t\tlstsq\t\terror")
rel_error = []
rel_error_lstsq = []
for i in range(trials_n):
    rel_error.append([])
    rel_error_lstsq.append([])
    for j in range(len(configs_t)):
        predicted = predicted_times_HK[i*len(configs_t)+j]
        predicted_lstsq = np.dot(lstsq_A[i*len(configs_t)+j], lstsq_weights)
        actual = actual_times_all[i*len(configs_t)+j]
        rel_error[i].append((predicted-actual)/actual)
        rel_error_lstsq[i].append((predicted_lstsq-actual)/actual)
        print("%i\t%i\t%i\t%f\t%f\t%f\t%f\t%f" %
                            (nvals[i], configs_t[j][0], configs_t[j][1],
                            actual, predicted, rel_error[i][-1], predicted_lstsq,
                            rel_error_lstsq[i][-1]))

print("\nHong Kim relative error:")
print("\t", end='')
for config in configs_t:
    print("(%i,%i)\t\t" % (config[0], config[1]), end='')
print("")
for i in range(trials_n):
    print("%i\t" % (nvals[i]), end='')
    for j in range(len(configs_t)):
        print("%f\t" % (rel_error[i][j]), end='')
    print("")

print("\nLeast squares relative error:")
print("\t", end='')
for config in configs_t:
    print("(%i,%i)\t\t" % (config[0], config[1]), end='')
print("")
for i in range(trials_n):
    print("%i\t" % (nvals[i]), end='')
    for j in range(len(configs_t)):
        print("%f\t" % (rel_error_lstsq[i][j]), end='')
    print("")
"""

if __name__ == '__main__':
    main()

