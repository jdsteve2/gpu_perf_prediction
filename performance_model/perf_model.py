
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

    # threads_per_warp:           number of threads per warp
    # issue_cycles:               number of cycles to execute one instruction
    # sm_clock_freq:              clock frequency of SMs (GHz), renamed from "Freq"
    # mem_bandwidth:              bandwidth between DRAM and GPU cores (GB/s)
    # roundtrip_DRAM_access_latency:        DRAM access latency, renamed from Mem_LD (?cycles)
    # departure_del_coal:         delay between two coalesced mem trans (?cycles)
    # departure_del_uncoal:       delay between two uncoalesced mem trans (?cycles)
    # mem_trans_per_warp_coal:    number of coalsced mem trans per warp
    # mem_trans_per_warp_uncoal:  number of uncoalsced mem trans per warp

    def __init__(self, gpu_name):
        if (gpu_name == 'GTX280'):
            self.threads_per_warp = 32
            self.issue_cycles = 4  #?
            self.sm_clock_freq = 1.3
            self.mem_bandwidth = 141.7
            self.roundtrip_DRAM_access_latency = 450
            self.departure_del_coal = 4
            self.departure_del_uncoal = 40
            self.mem_trans_per_warp_coal = 1
            self.mem_trans_per_warp_uncoal = 5.7  #see technical report??
            self.SM_count = 30
            self.max_threads_per_SM = 1024
            self.max_blocks_per_SM = 8
        elif (gpu_name == 'FX5600'):
            self.threads_per_warp = 32  # Table 1
            self.issue_cycles = 4  # Table 1
            self.sm_clock_freq = 1.35  # Table 3
            self.mem_bandwidth = 76.8  # Table 3
            self.roundtrip_DRAM_access_latency = 420  # Table 6
            self.departure_del_coal = 4  # Table 6
            self.departure_del_uncoal = 10  # Table 6
            self.mem_trans_per_warp_coal = 1  # Table 3
            self.mem_trans_per_warp_uncoal = 32  # Table 3
            self.SM_count = 16  # Table 3
            self.max_threads_per_SM = 768
            self.max_blocks_per_SM = 8
        elif (gpu_name == 'HKexample'):
            self.threads_per_warp = 32
            self.issue_cycles = 4
            self.sm_clock_freq = 1.0
            self.mem_bandwidth = 80
            self.roundtrip_DRAM_access_latency = 420
            self.departure_del_coal = 1
            self.departure_del_uncoal = 10
            self.mem_trans_per_warp_coal = 1
            self.mem_trans_per_warp_uncoal = 32
            self.SM_count = 16
            self.max_threads_per_SM = 1024
            self.max_blocks_per_SM = 8
        elif (gpu_name == 'TeslaK20'):
            self.threads_per_warp = 32
            self.issue_cycles = 4
            self.sm_clock_freq = 0.706
            self.mem_bandwidth = 208
            self.roundtrip_DRAM_access_latency = 230  # 230 from Kumar, 2014  TODO correct?
            self.departure_del_coal = 1  # from Krishnamani, Clemson U, 2014, for K20
            # TODO Is this^ correct?
            self.departure_del_uncoal = 38
            # TODO Is this^ correct? from Krishnamani, Clemson U, 2014, for K20
            self.mem_trans_per_warp_coal = 1  # TODO Is this correct?
            self.mem_trans_per_warp_uncoal = 32  # TODO check on this
            self.SM_count = 13
            self.max_threads_per_SM = 2048
            self.max_blocks_per_SM = 16
        else:
            print "Error: unknown hardware"
        #TODO use compute capability to get some of these numbers


class KernelStats(object):

    # comp_instructions:        total dynamic # of computation ins'ns per thread
    # mem_instructions_uncoal:  # of uncoalesced memory instructions per thread
    # mem_instructions_coal:    number of coalesced memory instructions per thread
    # synch_instructions:       total dynamic # of synch instructions per thread
    # mem_insns_total:         mem_instructions_uncoal + mem_instructions_coal
                        #TODO paper does not explain this, make sure it's correct
    # total_instructions:       comp_instructions + mem_insns_total

    def __init__(self, comp_instructions, mem_instructions_uncoal,
                        mem_instructions_coal, synch_instructions):
        self.comp_instructions = comp_instructions
        self.mem_instructions_uncoal = mem_instructions_uncoal
        self.mem_instructions_coal = mem_instructions_coal
        self.synch_instructions = synch_instructions
        self.mem_insns_total = mem_instructions_uncoal + mem_instructions_coal
        self.total_instructions = comp_instructions + self.mem_insns_total

    def __str__(self):
        return "\ncomp_insns: " + str(self.comp_instructions) + \
               "\nmem_insns_uncoal: " + str(self.mem_instructions_uncoal) + \
               "\nmem_insns_coal: " + str(self.mem_instructions_coal) + \
               "\nmem_insns_total: " + str(self.mem_insns_total) + \
               "\nsynch_insns: " + str(self.synch_instructions) + \
               "\ntotal_insns: " + str(self.total_instructions)


class ThreadConfig(object):
    def __init__(self, threads_per_block, blocks):
        self.threads_per_block = threads_per_block
        self.blocks = blocks


class PerfModel(object):

    def __init__(self, GPU_stats, kernel_stats, thread_config, dtype,
                                                    active_blocks=None):
        self.GPU_stats = GPU_stats
        self.kernel_stats = kernel_stats
        self.thread_config = thread_config
        data_size = dtype.itemsize

        # Calculate number of bytes loaded by full warp
        self.load_bytes_per_warp = GPU_stats.threads_per_warp * data_size

        # Determine # of blocks that can run simultaneously on one SM
        #TODO calculate this correctly figuring in register/shared mem usage
        if active_blocks is None:
            self.active_blocks_per_SM = min(
                float(GPU_stats.max_threads_per_SM)/thread_config.threads_per_block,
                GPU_stats.max_blocks_per_SM)
        else:
            self.active_blocks_per_SM = active_blocks
        #print("DEBUGGING... self.active_blocks_per_SM: ", self.active_blocks_per_SM)

        # Determine number of active SMs
        # active_SMs == SM_count, unless we have a very small number of blocks
        self.active_SMs = min(
                            float(thread_config.blocks)/self.active_blocks_per_SM,
                            GPU_stats.SM_count)

        # Calculate number of active warps per SM
        self.active_warps_per_SM = self.active_blocks_per_SM * \
                                   float(thread_config.threads_per_block)/ \
                                   GPU_stats.threads_per_warp

    def compute_total_cycles(self):

        # time (cycles) per warp spent on uncoalesced mem transactions
        mem_l_uncoal = self.GPU_stats.roundtrip_DRAM_access_latency + (
                       self.GPU_stats.mem_trans_per_warp_uncoal - 1) * \
                       self.GPU_stats.departure_del_uncoal

        # time (cycles) per warp spent on coalesced mem transactions
        mem_l_coal = self.GPU_stats.roundtrip_DRAM_access_latency

        # percent of mem transactions that are uncoalesced
        weight_uncoal = float(self.kernel_stats.mem_instructions_uncoal)/(
                        self.kernel_stats.mem_insns_total)

        # percent of mem transactions that are coalesced
        weight_coal = float(self.kernel_stats.mem_instructions_coal)/(
                        self.kernel_stats.mem_insns_total)

        # weighted average of mem latency (cycles) per warp
        mem_l = mem_l_uncoal * weight_uncoal + mem_l_coal * weight_coal

        # "minimum departure distance between two consecutive memory warps" -HK
        # (cycles)
        departure_delay = self.GPU_stats.departure_del_uncoal * \
                          self.GPU_stats.mem_trans_per_warp_uncoal * \
                          weight_uncoal + self.GPU_stats.departure_del_coal * \
                          weight_coal

        # "If the number of active warps is less than MWP_Without_BW_full,
        # the processor does not have enough number of warps to utilize 
        # memory level parallelism"
        mwp_without_bw_full = mem_l/departure_delay
        #mwp_without_bw_full = round(mwp_without_bw_full, 2)
        mwp_without_bw = min(mwp_without_bw_full, self.active_warps_per_SM)

        # memory cycles per warp
        mem_cycles = mem_l_uncoal * self.kernel_stats.mem_instructions_uncoal  \
                    + mem_l_coal * self.kernel_stats.mem_instructions_coal

        # computation cycles per warp
        comp_cycles = self.GPU_stats.issue_cycles * \
                      self.kernel_stats.total_instructions

        # active warps per SM TODO: forget n
        n = self.active_warps_per_SM

        # how many times does an SM execute active_blocks_per_SM blocks?
        reps_per_SM = float(self.thread_config.blocks)/(
                    self.active_blocks_per_SM * self.active_SMs)

        # bandwidth per warp (GB/second)
        bw_per_warp = self.GPU_stats.sm_clock_freq * \
                      float(self.load_bytes_per_warp)/mem_l
        #bw_per_warp = round(bw_per_warp, 3)

        # max memory warp parallelism (warps/SM) based on peak mem bandwidth
        mwp_peak_bw = float(self.GPU_stats.mem_bandwidth)/(
                      bw_per_warp * self.active_SMs)
        #mwp_peak_bw = round(mwp_peak_bw, 2)

        # Memory Warp Parallelism (MWP)
        # MWP: # of memory warps per SM that can be handled during mem_L cycles
        # MWP is minimum of three quantities:
        #  mwp_peak_bw: maximum number of warps based on peak mem bandwidth
        #  mwp_without_bw: if peak bw not reached, MWP is function of mem_l and departure_delay
        #  n: maximum number of active warps per SM based on machine resources like register usage, shared memory usage, etc.
        self.MWP = min(mwp_without_bw, mwp_peak_bw, n)  #TODO n already incorporated above
        #self.MWP = round(self.MWP, 2)

        # total cycles (per warp) / computation cycles (per warp)  
        # = max computation warp parallelism
        cwp_full = float(mem_cycles + comp_cycles)/comp_cycles
        #cwp_full = round(cwp_full, 2)

        # CWP cannot be greater than the max number of active warps per SM
        self.CWP = min(cwp_full, n)

        if (self.MWP == n) and (self.CWP == n):
            exec_cycles_app = (mem_cycles + comp_cycles +
                              float(comp_cycles)/self.kernel_stats.mem_insns_total*
                              (self.MWP-1))*reps_per_SM
        elif (self.CWP >= self.MWP) or (comp_cycles > mem_cycles):
            exec_cycles_app = (mem_cycles * float(n)/self.MWP +
                              float(comp_cycles)/self.kernel_stats.mem_insns_total*
                              (self.MWP-1))*reps_per_SM
        else:  # (self.MWP > self.CWP)
            exec_cycles_app = (mem_l + comp_cycles * n)*reps_per_SM

        # compute cost of synchronization instructions
        synch_cost = departure_delay * (self.MWP-1) *  \
                     self.kernel_stats.synch_instructions * \
                     self.active_blocks_per_SM*reps_per_SM

        # compute CPI (cycles per instruction) just to see what it is
        self.CPI = exec_cycles_app/(self.kernel_stats.total_instructions*(self.thread_config.threads_per_block/self.GPU_stats.threads_per_warp)*(self.thread_config.blocks/self.active_SMs))
        self.occ = n*self.GPU_stats.threads_per_warp/self.GPU_stats.max_threads_per_SM
        '''
        print "<debugging> mem_ld: ", self.GPU_stats.roundtrip_DRAM_access_latency
        print "<debugging> departure_del_uncoal: ", self.GPU_stats.departure_del_uncoal
        print "<debugging> threads_per_block: ", self.thread_config.threads_per_block
        print "<debugging> blocks: ", self.thread_config.blocks
        print "<debugging> active_blocks_per_sm: ", self.active_blocks_per_SM
        print "<debugging> active_sms: ", self.active_SMs
        print "<debugging> active_warps_per_sm: ", self.active_warps_per_SM
        print "<debugging> comp_insts: ", self.kernel_stats.comp_instructions
        print "<debugging> uncoal_mem_insts: ", self.kernel_stats.mem_instructions_uncoal
        print "<debugging> coal_mem_insts: ", self.kernel_stats.mem_instructions_coal
        print "<debugging> synch_insts: ", self.kernel_stats.synch_instructions
        print "<debugging> mem_trans_per_warp_coal: ", self.GPU_stats.mem_trans_per_warp_coal
        print "<debugging> mem_trans_per_warp_uncoal: ", self.GPU_stats.mem_trans_per_warp_uncoal
        print "<debugging> load_bytes_per_warp: ", self.load_bytes_per_warp
        print "<debugging> departure_delay: ", departure_delay
        print "<debugging> mem_l: ", mem_l
        print "<debugging> mwp_without_bw_full: ", mwp_without_bw_full
        print "<debugging> bw_per_warp: ", bw_per_warp
        print "<debugging> mwp_peak_bw: ", mwp_peak_bw
        print "<debugging> MWP: ", self.MWP
        print "<debugging> comp_cycles: ", comp_cycles
        print "<debugging> mem_cycles: ", mem_cycles
        print "<debugging> CWP_full: ", cwp_full
        print "<debugging> CWP: ", self.CWP
        print "<debugging> rep: ", reps_per_SM
        print "<debugging> exec_cycles_app: ", exec_cycles_app
        print "<debugging> synch_cost: ", synch_cost
        print "<debugging> CPI: ", CPI
        '''
        return exec_cycles_app+synch_cost


