from __future__ import division, print_function

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom  # noqa
from loopy.statistics import get_op_poly, get_DRAM_access_poly, get_barrier_poly
import sys
sys.path.append("../performance_model")
sys.path.append("../utils")
from perf_model import GPUStats, KernelStats, ThreadConfig, PerfModel
from utils import *
import math
import copy

run_mm = False
run_axpy = False
run_tp = False
run_conv = False
run_empt = False
run_fd = False

run_mm = True
run_axpy = True
run_tp = True
run_conv = True
run_empt = True
run_fd = True

warm_up_gpu = False
compute_const_manually = False
averaging_trials = 5
warmup_trials = 2


def main():
    # setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
    #print_device_info(ctx)

    HK_predict_all = []
    actual_times_all = []
    Atrain = []
    ytrain = []
    Atest = []
    ytest = []
    trials_n = 4

    if run_mm:
        nvals = [2**(9+x) for x in range(trials_n)]
        configs_t = [(8, 8), (16, 16), (24, 24), (32, 32)]
        run_mm_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                      actual_times_all, HK_predict_all, 'split', 'allcoal')
        '''
        A_mm2, HK_predict_mm2, actual_mm2 = run_mm_trials(ctx, queue, nvals,
                                                            configs_t, "partcoal")
        update_results(lstsq_A, A_mm2, HK_predict_all,
                       HK_predict_mm2, actual_times_all, actual_mm2)
        '''
    if run_axpy:
        nvals = [2**(25+x) for x in range(trials_n)]
        configs_t = [(64, 1), (128, 1), (256, 1), (512, 1), (1024, 1)]
        # TODO figure out problem with 64, 1 (why not slower?)
        run_axpy_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                        actual_times_all, HK_predict_all, 'split')
    if run_tp:
        nvals = [2**(10+x) for x in range(trials_n)]
        configs_t = [(8, 8), (16, 16), (24, 24), (32, 32)]
        run_tp_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                      actual_times_all, HK_predict_all, 'split', prefetch=True)
        run_tp_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                      actual_times_all, HK_predict_all, 'split', prefetch=False)
    if run_conv:
        nvals = [2**(8+x) for x in range(trials_n+1)]
        configs_t = [(8, 8), (16, 16), (32, 32)]
        run_conv_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                        actual_times_all, HK_predict_all, 'split')
    if run_empt:
        nvals = [2**(10+x) for x in range(trials_n)]
        configs_t = [(8, 8), (16, 16), (24, 24), (32, 32)]
        run_empt_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                        actual_times_all, HK_predict_all, 'split')
    if run_fd:
        nvals = [2**(10+x) for x in range(trials_n)]
        configs_t = [(8, 8), (16, 16), (24, 24), (32, 32)]
        run_fd_trials(ctx, queue, nvals, configs_t, Atrain, Atest, ytrain, ytest,
                        actual_times_all, HK_predict_all, 'split')

    # least squares calculations
    if run_mm or run_axpy or run_tp or run_conv or run_empt or run_fd:

        # TODO, make copies or move pointers?
        #TODO figure out when I really need copy.deepcopy

        # divide by runtime to minimize relative error rather than absolute error
        ones = np.ones(len(ytrain))
        Atrain_for_relerr = copy.deepcopy(Atrain)
        divide_rows_by_weights(Atrain_for_relerr, ytrain)

        lstsq_weights, resid, q, q = np.linalg.lstsq(Atrain_for_relerr, ones)
        U, s, V = np.linalg.svd(Atrain_for_relerr, full_matrices=False)
        cos_angles = get_cos_angles_bt_rows(Atrain_for_relerr)

        #print("Least Squares Residual:\n", np.dot(Atrain, lstsq_weights)-ytrain)
        print("Least Squares Residual:\n",
              np.dot(Atrain_for_relerr, lstsq_weights)-ytrain)
        print("Least Squares singular values:\n", s)
        print("="*40+"TIMING RESULTS")

        # print least squares results
        print("i\tactual\t\tlstsq\t\terror")
        rel_error_lstsq = []
        for i in range(len(ytest)):
            predicted_lstsq = np.dot(Atest[i], lstsq_weights)
            actual = ytest[i]
            rel_error_lstsq.append((predicted_lstsq-actual)/actual)
            print("%i\t%.7f\t%.7f\t%.7f" %
                  (i, actual, predicted_lstsq, rel_error_lstsq[i]))

        # print hong kim results
        print("i\tactual\t\tHK\t\terror")
        rel_error_HK = []
        for i in range(len(HK_predict_all)):
            predicted = HK_predict_all[i]
            actual = actual_times_all[i]
            rel_error_HK.append((predicted-actual)/actual)
            print("%i\t%.7f\t%.7f\t%.7f" % (i, actual, predicted, rel_error_HK[i]))

        print("avg relative error HK: ", np.average(np.absolute(rel_error_HK)))
        print("avg relative error LS: ", np.average(np.absolute(rel_error_lstsq)))
        print("med relative error LS: ", np.median(np.absolute(rel_error_lstsq)))

        print(np.average(cos_angles))
        print_Ay(Atrain, ytrain)

        print("Weights:")
        for item in lstsq_weights:
            print("\t%e" % (item), end='')
        print()


def update_LS_matrix(A, flops, intops,
                     f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s,
                     barrier_ct, blocks, thread_work_units, itemsize, model):

    reps_per_SM = math.ceil(blocks/(model.active_blocks_per_SM *
                                    model.GPU_stats.SM_count))
    multiplier = reps_per_SM
    # TODO assumes there are enough blocks to fully load all SMs
    A.append([multiplier*itemsize*flops/thread_work_units,
              multiplier*itemsize*intops/thread_work_units,
              multiplier*itemsize*f32uncoal_l/thread_work_units,
              multiplier*itemsize*f32coal_l/thread_work_units,
              multiplier*itemsize*f32uncoal_s/thread_work_units,
              multiplier*itemsize*f32coal_s/thread_work_units,
              multiplier*itemsize*min(f32uncoal_s, f32uncoal_l)/thread_work_units,
              multiplier*itemsize*min(f32coal_s, f32coal_l)/thread_work_units,
              multiplier*barrier_ct])
    if not compute_const_manually:
        A[-1].append(1.0)


def split_for_train_test(A, y):

    Atrain = []
    Atest = []
    ytrain = []
    ytest = []

    for row in range(len(A)):
        '''
        if row < len(A)-16:
            Atrain.append(copy.deepcopy(A[row]))
            ytrain.append(copy.deepcopy(y[row]))
        else:
            Atest.append(copy.deepcopy(A[row]))
            ytest.append(copy.deepcopy(y[row]))
        '''
        '''
        if row % 2 == 0:
            Atrain.append(copy.deepcopy(A[row]))
            ytrain.append(copy.deepcopy(y[row]))
        else:
            Atest.append(copy.deepcopy(A[row]))
            ytest.append(copy.deepcopy(y[row]))

        '''
        Atrain.append(copy.deepcopy(A[row]))
        ytrain.append(copy.deepcopy(y[row]))
        Atest.append(copy.deepcopy(A[row]))
        ytest.append(copy.deepcopy(y[row]))
        #'''
    return (Atrain, ytrain, Atest, ytest)


def run_mm_trials(ctx, queue, nvals, configs_t,
                  Atrain_all, Atest_all, ytrain_all, ytest_all,
                  actual_times_all, HK_predict_all, train_test_config, version):

    A = []
    HK_predict = []
    actual = []
    dtype = np.float32

    #TODO figure out smem usage issue
    for n in nvals:
        a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=dtype)
        b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=dtype)
        c_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=dtype)

        order = "C"
        knl = lp.make_kernel(
            "{[i,j,k]: 0<=i,j,k<%d}" % n,
            [
                "c[i, j] = sum(k, a[i, k]*b[k, j])"
            ], [
                lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                lp.GlobalArg("c", dtype, shape=(n, n), order=order),
            ], name="matmul")

        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            if version == "allcoal":
                knl = lp.split_iname(knl, "i", BSIZEy,
                                     outer_tag="g.0", inner_tag="l.1")
                knl = lp.split_iname(knl, "j", BSIZEx,
                                     outer_tag="g.1", inner_tag="l.0")
            elif version == "partcoal":
                knl = lp.split_iname(knl, "i", BSIZEy,
                                     outer_tag="g.0", inner_tag="l.0")
                knl = lp.split_iname(knl, "j", BSIZEx,
                                     outer_tag="g.1", inner_tag="l.1")
            else:
                1/0
                # TODO error
            ksplit = BSIZEy
            knl = lp.split_iname(knl, "k", ksplit)
            knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"])
            knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner", ])

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True)
            #print "Correctness check: \n", check
            # use ptx src to determine resource usage

            #ptx_dump(ctx, knl, n, BSIZEx, BSIZEy)

            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict({'n': n})

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, {'n': n})

            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(
                                                                  sub_map, {'n': n})
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

            trial_times = []
            for i in range(averaging_trials+warmup_trials):
                evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev, c=c_mat_dev)
                evt.wait()
                trial_times.append((evt.profile.END - evt.profile.START)*1e-9)
            avg_time = np.average(trial_times[warmup_trials:])

            gstats = GPUStats('TeslaK20')
            if BSIZEx == 8 or BSIZEx == 32:  # TODO fix hack
                reg32_per_thread = 25
            elif BSIZEx == 24:
                reg32_per_thread = 18
            elif BSIZEx == 16:
                reg32_per_thread = 22

            shared_mem_per_block = 4*ksplit*(BSIZEx+BSIZEy)
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy  # TODO never used
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_ct, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(dtype))
            cycles = model.compute_total_cycles()
            actual.append(avg_time)
            HK_predict.append(cycles/(gstats.sm_clock_freq*10**9))

            '''
            print "actual runtime: ", actual[-1]
            print "total predicted time: ", predicted[-1]
            print "total predicted execution cycles: ", cycles
            print "="*40
            '''
            update_LS_matrix(A, flops, iops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(dtype).itemsize, model)

    if train_test_config == 'split':
        Atrain, ytrain, Atest, ytest = split_for_train_test(A, actual)
        append_mats([Atrain_all, Atest_all, ytrain_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [Atrain, Atest, ytrain, ytest,
                     actual, HK_predict])
    if train_test_config == 'train':
        append_mats([Atrain_all, ytrain_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])
    if train_test_config == 'test':
        append_mats([Atest_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])


def run_axpy_trials(ctx, queue, nvals, configs_t,
                    Atrain_all, Atest_all, ytrain_all, ytest_all,
                    actual_times_all, HK_predict_all, train_test_config):

    A = []
    HK_predict = []
    actual = []
    dtype = np.float32

    #TODO figure out smem usage issue
    for n in nvals:
        x_vec_dev = cl.clrandom.rand(queue, n, dtype=dtype)
        y_vec_dev = cl.clrandom.rand(queue, n, dtype=dtype)
        z_vec_dev = cl.clrandom.rand(queue, n, dtype=dtype)

        knl = lp.make_kernel(
            "[n] -> {[i]: 0<=i<%d}" % n,
            [
                "z[i] = 5.0*x[i]+7.0*y[i]"
            ], [
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
            #ptx_dump(ctx, knl, n, BSIZEx, BSIZEy)

            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict({'n': n})
            op_map = get_op_poly(knl)

            flops, iops = get_32b_ops(op_map, {'n': n})
            sub_map = get_DRAM_access_poly(knl)  # noqa

            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(
                                                                  sub_map, {'n': n})
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

            trial_times = []
            for i in range(averaging_trials+warmup_trials):
                evt, (out,) = knl(queue, x=x_vec_dev, y=y_vec_dev, z=z_vec_dev)
                evt.wait()
                trial_times.append((evt.profile.END - evt.profile.START)*1e-9)
            avg_time = np.average(trial_times[warmup_trials:])

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 20
            shared_mem_per_block = 0
            total_blocks = math.ceil(n/(BSIZEx*unroll))
            kstats = KernelStats(flops*unroll/n, f32uncoal*unroll/n,
                                 f32coal*unroll/n, barrier_ct, reg32_per_thread,
                                 shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig, np.dtype(dtype))
            cycles = model.compute_total_cycles()

            actual.append(avg_time)
            HK_predict.append(cycles/(gstats.sm_clock_freq*10**9))

            update_LS_matrix(A, flops, iops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n/unroll,
                             np.dtype(dtype).itemsize, model)

    if train_test_config == 'split':
        Atrain, ytrain, Atest, ytest = split_for_train_test(A, actual)
        append_mats([Atrain_all, Atest_all, ytrain_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [Atrain, Atest, ytrain, ytest,
                     actual, HK_predict])
    if train_test_config == 'train':
        append_mats([Atrain_all, ytrain_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])
    if train_test_config == 'test':
        append_mats([Atest_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])


def run_tp_trials(ctx, queue, nvals, configs_t,
                  Atrain_all, Atest_all, ytrain_all, ytest_all, actual_times_all,
                  HK_predict_all, train_test_config, prefetch=True):

    A = []
    HK_predict = []
    actual = []
    dtype = np.float32

    for n in nvals:
        a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=dtype)
        b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=dtype)
        order = "C"
        knl = lp.make_kernel(
                "{[i,j]: 0<=i,j<%d}" % n,
                [
                    "b[i, j] = a[j, i]"
                ], [
                    lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                    lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                ],
                name="transpose")

        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            knl = lp.split_iname(knl, "i", BSIZEy, outer_tag="g.0", inner_tag="l.1")
            knl = lp.split_iname(knl, "j", BSIZEx, outer_tag="g.1", inner_tag="l.0")
            if prefetch:
                knl = lp.add_prefetch(knl, 'a', ["i_inner", "j_inner"])

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True)
            #print "Correctness check: \n", check

            # use ptx src to determine resource usage
            #ptx_dump(ctx, knl, n, BSIZEx, BSIZEy)

            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict({'n': n})

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, {'n': n})

            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(
                                                                  sub_map, {'n': n})
            f32coal = f32coal_l + f32coal_s
            f32uncoal = f32uncoal_l + f32uncoal_s
            # execute
            # -------
            #print "="*40+"TIMING RESULTS"
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)
            #if not prefetch:
            #    knl = lp.set_options(knl, write_cl=True, highlight_cl=True)

            trial_times = []
            for i in range(averaging_trials+warmup_trials):
                evt, (out,) = knl(queue, a=a_mat_dev, b=b_mat_dev)
                evt.wait()
                trial_times.append((evt.profile.END - evt.profile.START)*1e-9)
            avg_time = np.average(trial_times[warmup_trials:])
            #if not prefetch:
            #    1/0
            gstats = GPUStats('TeslaK20')
            if n % BSIZEx == 0 and n % BSIZEy == 0:
                if prefetch:
                    reg32_per_thread = 10
                else:
                    reg32_per_thread = 8
            else:
                if prefetch:
                    reg32_per_thread = 8
                else:
                    reg32_per_thread = 9

            if prefetch:
                shared_mem_per_block = 4*BSIZEx*BSIZEy
            else:
                shared_mem_per_block = 0
            # TODO why is HK way  off on the non-prefetch version?
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy  # TODO unused
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_ct, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(dtype))
            cycles = model.compute_total_cycles()

            actual.append(avg_time)
            HK_predict.append(cycles/(gstats.sm_clock_freq*10**9))

            #update_LS_matrix(A, flops, f32coal_l, f32coal_s, f32uncoal_l,
            update_LS_matrix(A, flops, iops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(dtype).itemsize, model)

    if train_test_config == 'split':
        Atrain, ytrain, Atest, ytest = split_for_train_test(A, actual)
        append_mats([Atrain_all, Atest_all, ytrain_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [Atrain, Atest, ytrain, ytest,
                     actual, HK_predict])
    if train_test_config == 'train':
        append_mats([Atrain_all, ytrain_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])
    if train_test_config == 'test':
        append_mats([Atest_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])


def run_conv_trials(ctx, queue, nvals, configs_t,
                    Atrain_all, Atest_all, ytrain_all, ytest_all,
                    actual_times_all, HK_predict_all, train_test_config):

    A = []
    HK_predict = []
    actual = []
    dtype = np.float32
    ncolors = 3

    for n in nvals:
        knl = lp.make_kernel(
            "{ [iimg, ifeat, icolor, im_x, im_y, f_x, f_y]: \
                -f_w <= f_x,f_y <= f_w \
                and 0 <= im_x < im_w and 0 <= im_y < im_h \
                and 0<=iimg<=nimgs and 0<=ifeat<nfeats and 0<=icolor<ncolors \
             }",
            """
            out[iimg, ifeat, im_x, im_y] = sum((f_x, f_y, icolor), \
                img[iimg, f_w+im_x-f_x, f_w+im_y-f_y, icolor] \
                * f[ifeat, f_w+f_x, f_w+f_y, icolor])
            """,
            [
                lp.GlobalArg("f", dtype, shape=lp.auto),
                lp.GlobalArg("img", dtype, shape=lp.auto),
                lp.GlobalArg("out", dtype, shape=lp.auto),
                "..."
            ],
            assumptions="f_w>=1 and im_w, im_h >= 2*f_w+1 and nfeats>=1 and nimgs>=0",
            flags="annotate_inames",
            defines=dict(ncolors=ncolors),
            name="conv")

        f_w = 3
        knl = lp.fix_parameters(knl, f_w=f_w)
        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl

            im_w = n
            im_h = n
            nfeats = 3
            nimgs = 3
            f_dev = cl.clrandom.rand(queue, (nfeats, 2*f_w+1, 2*f_w+1, ncolors),
                                     dtype=dtype)
            img_dev = cl.clrandom.rand(queue, (nimgs+1, n+2*f_w, n+2*f_w, ncolors),
                                       dtype=dtype)

            knl = lp.split_iname(knl, "im_x", BSIZEx,
                                 outer_tag="g.0", inner_tag="l.0")
            knl = lp.split_iname(knl, "im_y", BSIZEy,
                                 outer_tag="g.1", inner_tag="l.1")
            knl = lp.tag_inames(knl, dict(ifeat="g.2"))
            knl = lp.add_prefetch(knl, "f[ifeat,:,:,:]")
            knl = lp.add_prefetch(knl, "img", "im_x_inner, im_y_inner, f_x, f_y")

            params = dict(im_w=im_w, im_h=im_h, f_w=f_w, nfeats=nfeats, nimgs=nimgs)

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True,
            #                            parameters=params)
            #print "Correctness check: \n", check
            # use ptx src to determine resource usage
            #ptx_dump(ctx, knl, n, BSIZEx, BSIZEy)

            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict(params)

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, params)
            #TODO why do blk sizes that don't fit perfecty increase total flops/iops
            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(
                                                                    sub_map, params)
            f32coal = f32coal_l + f32coal_s
            f32uncoal = f32uncoal_l + f32uncoal_s

            # execute
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)

            trial_times = []
            for i in range(averaging_trials+warmup_trials):
                evt, (out,) = knl(queue, f=f_dev, img=img_dev, im_w=im_w, im_h=im_h,
                                  nfeats=nfeats, nimgs=nimgs)
                evt.wait()
                trial_times.append((evt.profile.END - evt.profile.START)*1e-9)
            avg_time = np.average(trial_times[warmup_trials:])

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 33
            shared_mem_per_block = (ncolors * (f_w*2+1) * (f_w*2+1) +
                                    (BSIZEx+f_w*2) * (BSIZEy+f_w*2)
                                    ) * np.dtype(dtype).itemsize
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy  # TODO unused
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_ct, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig, np.dtype(dtype))
            cycles = model.compute_total_cycles()

            actual.append(avg_time)
            HK_predict.append(cycles/(gstats.sm_clock_freq*10**9))
            update_LS_matrix(A, flops, iops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(dtype).itemsize, model)
            #TODO try total_threads for n*n

    if train_test_config == 'split':
        Atrain, ytrain, Atest, ytest = split_for_train_test(A, actual)
        append_mats([Atrain_all, Atest_all, ytrain_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [Atrain, Atest, ytrain, ytest,
                     actual, HK_predict])
    if train_test_config == 'train':
        append_mats([Atrain_all, ytrain_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])
    if train_test_config == 'test':
        append_mats([Atest_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])


def run_empt_trials(ctx, queue, nvals, configs_t,
                    Atrain_all, Atest_all, ytrain_all, ytest_all,
                    actual_times_all, HK_predict_all, train_test_config):

    A = []
    HK_predict = []
    actual = []
    dtype = np.float32

    for n in nvals:
        knl = lp.make_kernel(
                "{[i,j]: 0<=i,j<%d}" % n,
                [
                    ""
                ],
                name="empty")

        for BSIZEx, BSIZEy in configs_t:

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, print_code=True)
            #print "Correctness check: \n", check

            # use ptx src to determine resource usage
            #ptx_dump(ctx, knl, n, BSIZEx, BSIZEy)

            params = {'n': n}
            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict(params)

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, params)

            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(
                                                                    sub_map, params)

            # execute
            # -------
            #print "="*40+"TIMING RESULTS"
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)

            trial_times = []
            for i in range(averaging_trials+warmup_trials):
                evt, out = knl(queue)
                evt.wait()
                trial_times.append((evt.profile.END - evt.profile.START)*1e-9)
            avg_time = np.average(trial_times[warmup_trials:])

            gstats = GPUStats('TeslaK20')
            reg32_per_thread = 2
            shared_mem_per_block = 0
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy  # TODO unused
            # TODO actually increase threads/blocks but expect 0 result
            kstats = KernelStats(0, 0, 0, barrier_ct, reg32_per_thread,
                                 shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(dtype))
            cycles = model.compute_total_cycles()

            actual.append(avg_time)
            HK_predict.append(cycles/(gstats.sm_clock_freq*10**9))

            update_LS_matrix(A, flops, iops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(dtype).itemsize, model)

    if train_test_config == 'split':
        Atrain, ytrain, Atest, ytest = split_for_train_test(A, actual)
        append_mats([Atrain_all, Atest_all, ytrain_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [Atrain, Atest, ytrain, ytest,
                     actual, HK_predict])
    if train_test_config == 'train':
        append_mats([Atrain_all, ytrain_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])
    if train_test_config == 'test':
        append_mats([Atest_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])


def run_fd_trials(ctx, queue, nvals, configs_t,
                  Atrain_all, Atest_all, ytrain_all, ytest_all,
                  actual_times_all, HK_predict_all, train_test_config):

    A = []
    HK_predict = []
    actual = []
    dtype = np.float32

    for n in nvals:
        u_mat_dev = cl.clrandom.rand(queue, (n+2, n+2), dtype=dtype)
        knl = lp.make_kernel(
              "{[i,j]: 0<=i,j<n}",
              "result[i,j] = u[i, j]**2 + -1 + (-4)*u[i + 1, j + 1] \
                    + u[i + 1 + 1, j + 1] + u[i + 1 + -1, j + 1] \
                    + u[i + 1, j + 1 + 1] + u[i + 1, j + 1 + -1]",
              name="finite_diff")
        knl = lp.add_and_infer_dtypes(knl, {"u": dtype})
        ref_knl = knl

        for BSIZEx, BSIZEy in configs_t:

            knl = ref_knl
            knl = lp.split_iname(knl,
                    "i", BSIZEx, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl,
                    "j", BSIZEy, outer_tag="g.0", inner_tag="l.0")
            knl = lp.add_prefetch(knl, "u",
                    ["i_inner", "j_inner"],
                    fetch_bounding_box=True)

            #check = lp.auto_test_vs_ref(ref_knl, ctx, knl, parameters=dict(n=n),
            #                            print_code=True)
            #print "Correctness check: \n", check

            # use ptx src to determine resource usage
            #ptx_dump(ctx, knl, n, BSIZEx, BSIZEy)

            params = {'n': n}
            barrier_poly = get_barrier_poly(knl)
            barrier_ct = barrier_poly.eval_with_dict(params)

            op_map = get_op_poly(knl)
            flops, iops = get_32b_ops(op_map, params)

            sub_map = get_DRAM_access_poly(knl)  # noqa
            f32coal_l, f32coal_s, f32uncoal_l, f32uncoal_s = get_DRAM_f32_accesses(
                                                                    sub_map, params)
            f32coal = f32coal_l + f32coal_s
            f32uncoal = f32uncoal_l + f32uncoal_s

            # execute
            # -------
            #print "="*40+"TIMING RESULTS"
            print("running kernel...")
            #knl = lp.set_options(knl, write_cl=True, highlight_cl=True)

            trial_times = []
            for i in range(averaging_trials+warmup_trials):
                evt, (out,) = knl(queue, u=u_mat_dev)
                evt.wait()
                trial_times.append((evt.profile.END - evt.profile.START)*1e-9)
            avg_time = np.average(trial_times[warmup_trials:])

            gstats = GPUStats('TeslaK20')
            if n % BSIZEx == 0 and n % BSIZEy == 0:
                reg32_per_thread = 14
            else:
                reg32_per_thread = 16

            shared_mem_per_block = 4*(BSIZEx+2)*(BSIZEy+2)
            total_blocks = math.ceil(n/BSIZEx)*math.ceil(n/BSIZEy)
            total_threads = total_blocks*BSIZEx*BSIZEy  # TODO unused
            kstats = KernelStats(flops/(n*n), f32uncoal/(n*n), f32coal/(n*n),
                                 barrier_ct, reg32_per_thread, shared_mem_per_block)
            tconfig = ThreadConfig(BSIZEx*BSIZEy, total_blocks)
            model = PerfModel(gstats, kstats, tconfig,
                            np.dtype(dtype))
            cycles = model.compute_total_cycles()

            actual.append(avg_time)
            HK_predict.append(cycles/(gstats.sm_clock_freq*10**9))

            update_LS_matrix(A, flops, iops, f32coal_l, f32coal_s, f32uncoal_l,
                             f32uncoal_s, barrier_ct, total_blocks, n*n,
                             np.dtype(dtype).itemsize, model)

    if train_test_config == 'split':
        Atrain, ytrain, Atest, ytest = split_for_train_test(A, actual)
        append_mats([Atrain_all, Atest_all, ytrain_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [Atrain, Atest, ytrain, ytest,
                     actual, HK_predict])
    if train_test_config == 'train':
        append_mats([Atrain_all, ytrain_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])
    if train_test_config == 'test':
        append_mats([Atest_all, ytest_all,
                     actual_times_all, HK_predict_all],
                    [A, actual,
                     actual, HK_predict])

if __name__ == '__main__':
    main()

