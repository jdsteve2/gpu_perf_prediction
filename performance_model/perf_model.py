
from __future__ import division

__copyright__ = "Copyright (C) 2015 James Stevens"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Citations #
# Initial performance model is from the following paper:
# Hong, Kim, 2009,
#       "An analytical model for a gpu architecture
#        with memory-level and thread-level parallelism awareness,"

# parameters from Table 1 in Hong Kim paper

class GPUStats(object):

    # threads_per_warp:             number of threads per warp
    # issue_cycles:                 number of cycles to execute one instruction
    # sm_clock_freq:                clock frequency of the SMs, renamed from "Freq"
    # mem_bandwidth:                bandwidth between DRAM and GPU cores
    # DRAM_access_latency:          DRAM access latency, renamed from "Mem_LD"
    # departure_del_coal:           delay between two coalesced mem transactions (cycles?)#TODO
    # departure_del_uncoal:         delay between two uncoalesced mem transactions (cycles?)#TODO
    # mem_trans_per_warp_coal:      number of mem transactions per warp (coalesced)
    # mem_trans_per_warp_uncoal:    number of mem transactions per warp (uncoalsced)

    def __init__(self,gpu_name):
        if (gpu_name == 'GTX280'):
            self.threads_per_warp = 32
            self.issue_cycles = 4
            self.sm_clock_freq = 1.3
            self.mem_bandwidth = 141.7
            self.DRAM_access_latency = 450
            self.departure_del_coal = 4
            self.departure_del_uncoal = 40
            self.mem_trans_per_warp_coal = 1
            self.mem_trans_per_warp_uncoal = 12
            self.SM_count = 30
            self.max_threads_per_SM = 1024
            self.max_blocks_per_SM = 8
        elif (gpu_name == 'HKexample'):
            self.threads_per_warp = 32
            self.issue_cycles = 4
            self.sm_clock_freq = 1.0  #
            self.mem_bandwidth = 80  #
            self.DRAM_access_latency = 420  #
            self.departure_del_coal = 1  #
            self.departure_del_uncoal = 10  #
            self.mem_trans_per_warp_coal = 1  #
            self.mem_trans_per_warp_uncoal = 32  #
            self.SM_count = 16  #
            self.max_threads_per_SM = 1024
            self.max_blocks_per_SM = 8
        elif (gpu_name == 'TeslaK20'):
            self.threads_per_warp = 32
            self.issue_cycles = 4
            self.sm_clock_freq = 0.706  #
            self.mem_bandwidth = 208  #
            self.DRAM_access_latency = 230  # from Kumar, 2014
            self.departure_del_coal = 1  # TODO Is this correct?
            self.departure_del_uncoal = 10  # TODO Is this correct?
            self.mem_trans_per_warp_coal = 1  # TODO Is this correct?
            self.mem_trans_per_warp_uncoal = 32  # TODO check on this
            self.SM_count = 13  #
            self.max_threads_per_SM = 2048
            self.max_blocks_per_SM = 16
        else:
            print "Error: unknown hardware"
        #TODO use compute capability to get some of these numbers


class KernelStats(object):

    # comp_instructions:        total dynamic # of computation ins'ns in 1 thread
    # mem_instructions_uncoal:  # of uncoalesced memory instructions in one thread
    # mem_instructions_coal:    number of coalesced memory instructions in one thread
    # synch_instructions:       total dynamic # of synch instructions in one thread
    # mem_instructions:         mem_instructions_uncoal + mem_instructions_coal
                                #TODO make sure this is correct
    # total_instructions:       comp_instructions + mem_instructions

    def __init__(self, comp_instructions, mem_instructions_uncoal,
                        mem_instructions_coal, synch_instructions):
        self.comp_instructions = comp_instructions
        self.mem_instructions_uncoal = mem_instructions_uncoal
        self.mem_instructions_coal = mem_instructions_coal
        self.synch_instructions = synch_instructions
        self.mem_instructions = mem_instructions_uncoal + mem_instructions_coal
        self.total_instructions = comp_instructions + self.mem_instructions

    def __str__(self):
        return "\ncomp_insns: " + str(self.comp_instructions) + \
               "\nmem_insns_uncoal: " + str(self.mem_instructions_uncoal) + \
               "\nmem_insns_coal: " + str(self.mem_instructions_coal) + \
               "\nmem_insns_total: " + str(self.mem_instructions) + \
               "\nsynch_insns: " + str(self.synch_instructions) + \
               "\ntotal_insns: " + str(self.total_instructions)


class ThreadConfig(object):
    def __init__(self, threads_per_block, blocks):
        self.threads_per_block = threads_per_block
        self.blocks = blocks


class PerfModel(object):

    def __init__(self, GPU_stats, kernel_stats, thread_config, dtype):

        self.GPU_stats = GPU_stats
        self.kernel_stats = kernel_stats
        self.thread_config = thread_config
	data_size = dtype.itemsize

        self.load_bytes_per_warp = GPU_stats.threads_per_warp * data_size

        #TODO calculate this correctly figuring in register/shared mem usage
        self.active_blocks_per_SM = min(
                GPU_stats.max_threads_per_SM/thread_config.threads_per_block,
                GPU_stats.max_blocks_per_SM)
        print("DEBUGGING... self.active_blocks_per_SM: ",self.active_blocks_per_SM)
        #self.active_blocks_per_SM = 5  # TODO
        self.active_SMs = min(
                            thread_config.blocks/self.active_blocks_per_SM,
                            GPU_stats.SM_count)
        print("DEBUGGING... self.active_SMs: ",self.active_SMs)
        self.active_warps_per_SM = self.active_blocks_per_SM * \
                    thread_config.threads_per_block/GPU_stats.threads_per_warp

    def compute_exec_cycles(self):

        mem_l_uncoal = self.GPU_stats.DRAM_access_latency + (self.GPU_stats.mem_trans_per_warp_uncoal - 1)  \
                        * self.GPU_stats.departure_del_uncoal
        mem_l_coal = self.GPU_stats.DRAM_access_latency
        weight_uncoal = self.kernel_stats.mem_instructions_uncoal/(
                        self.kernel_stats.mem_instructions_uncoal +
                        self.kernel_stats.mem_instructions_coal)
        weight_coal = self.kernel_stats.mem_instructions_coal/(
                      self.kernel_stats.mem_instructions_coal +
                      self.kernel_stats.mem_instructions_uncoal)
        mem_l = mem_l_uncoal * weight_uncoal + mem_l_coal * weight_coal
        departure_delay = self.GPU_stats.departure_del_uncoal * \
                          self.GPU_stats.mem_trans_per_warp_uncoal * \
                          weight_uncoal + self.GPU_stats.departure_del_coal * \
                          weight_coal
        mwp_without_bw_full = mem_l/departure_delay
        mwp_without_bw = min(mwp_without_bw_full, self.active_warps_per_SM)
        mem_cycles = mem_l_uncoal * self.kernel_stats.mem_instructions_uncoal  \
                    + mem_l_coal * self.kernel_stats.mem_instructions_coal
        comp_cycles = self.GPU_stats.issue_cycles * \
                      self.kernel_stats.total_instructions
        n = self.active_warps_per_SM
        rep = self.thread_config.blocks/(self.active_blocks_per_SM * self.active_SMs)

        bw_per_warp = self.GPU_stats.sm_clock_freq*self.load_bytes_per_warp/mem_l
        mwp_peak_bw = self.GPU_stats.mem_bandwidth/(bw_per_warp * self.active_SMs)
        MWP = min(mwp_without_bw, mwp_peak_bw, n)

        cwp_full = (mem_cycles + comp_cycles)/comp_cycles
        CWP = min(cwp_full, n)

        if (MWP == n) and (CWP == n):  # TODO correct?
            exec_cycles_app = (mem_cycles + comp_cycles +
                               comp_cycles/self.kernel_stats.mem_instructions *
                               (MWP-1))*rep
        elif (CWP >= MWP) or (comp_cycles > mem_cycles):
            exec_cycles_app = (mem_cycles * n/MWP +
                               comp_cycles/self.kernel_stats.mem_instructions *
                               (MWP-1))*rep
        elif (MWP > CWP):
            exec_cycles_app = (mem_l + comp_cycles * n) * rep
        else:
            print "Error..."

        synch_cost = departure_delay * (MWP-1) *  \
                     self.kernel_stats.synch_instructions * \
                     self.active_blocks_per_SM*rep
        return exec_cycles_app+synch_cost






