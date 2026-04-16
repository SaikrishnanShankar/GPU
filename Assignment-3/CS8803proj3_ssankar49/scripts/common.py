import os

trace_dir = os.getenv('MACSIM_TRACE_DIR')

benchmarks = {
    'lavaMD_5':         f'{trace_dir}/lavaMD/5/kernel_config.txt',
    'nn_256k':          f'{trace_dir}/nn/256k/kernel_config.txt',
    'backprop_8192':    f'{trace_dir}/backprop/8192/kernel_config.txt',
    'crystal_q12':      f'{trace_dir}/crystal/q12/kernel_config.txt',
    'hotspot_r512h2i2': f'{trace_dir}/hotspot/r512h2i2/kernel_config.txt'
}

gpu_configs = {
    'RR':   'xmls/gpuconfig_8c_rr.xml',
    'GTO':  'xmls/gpuconfig_8c_gto.xml',
    'CCWS': 'xmls/gpuconfig_8c_ccws.xml'
}

stats = [
    'NUM_CYCLES',
    'NUM_INSTRS_RETIRED',
    'NUM_STALL_CYCLES',
    'NUM_MEM_REQUESTS',
    'NUM_MEM_RESPONSES',
    'AVG_RESPONSE_LATENCY',
    'NUM_TTIMEDOUT_REQUESTS',
    'INSTR_PER_CYCLE',
    'CACHE_NUM_ACCESSES',
    'CACHE_NUM_HITS',
    'CACHE_HIT_RATE_PERC',
    'MISSES_PER_1000_INSTR'
]