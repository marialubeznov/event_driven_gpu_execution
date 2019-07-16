import m5
from m5.objects import *

class MyCache(BaseCache):
    assoc = 2
#    block_size = 64
    hit_latency = '1'
    response_latency = '1'
    mshrs = 10
    tgts_per_mshr = 5

class MyL1Cache(MyCache):
    assoc = 4

cpu = TimingSimpleCPU(cpu_id = 0)
cpu.addTwoLevelCacheHierarchy(MyL1Cache(size = '128kB'),
                              MyL1Cache(size = '256kB'),
                              MyCache(size='2MB', hit_latency='10', response_latency='10'))

system = System(cpu = cpu,
        physmem = SimpleMemory(),
        membus = SystemXBar())


system.system_port = system.membus.slave
system.physmem.port = system.membus.master

cpu.createInterruptController()
cpu.connectAllPorts(system.membus)
system.clk_domain = SrcClockDomain()
system.clk_domain.clock = '1GHz'
system.clk_domain.voltage_domain = VoltageDomain()

#cpu.clock='2GHz'

root = Root(full_system=False, system = system)
root.system.cpu.workload = LiveProcess(cmd=['tests/test-progs/hello/bin/x86/linux/hello'])

m5.instantiate()

exit_event = m5.simulate(m5.MaxTick)
