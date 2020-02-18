#!/usr/bin/env python
from os import makedirs, stat, chmod
from stat import S_IEXEC
from datetime import datetime

# --------------- parameters
force_generate_job = False  # If True overwrites existing job if any
job_time = 10 # [min]
exe_str = 'v2_pool' # v1 v2 v1_pool v2_pool
arch = 'mc' # or 'gpu'
apex = 1 # or 0 for no APEX

len_m = 3000
len_n = 3000
len_k = 523000
tile_m = 512
tile_n = 512
tile_k = 10000
pgrid_rows = 5
pgrid_cols = 2
blk_rows = 512
blk_cols = 512

# set via cmake or manually
build_dir = "@PROJECT_BINARY_DIR@"
source_dir = "@PROJECT_SOURCE_DIR@"

# mc / gpu partition on daint
job_nprocs_per_node = 2 if arch == 'mc' else 1
job_threads_per_proc = 18 if arch == 'mc' else 12
nprocs = pgrid_rows * pgrid_cols
if nprocs % job_nprocs_per_node != 0:
    raise ValueError('The number of processes must be a multiple of 2!')
job_nodes = nprocs // job_nprocs_per_node
job_queue = "normal"

# create job dir in cwd
date_str = datetime.now().strftime('%Y.%m.%d')
job_name = ('{date_str}__'
            '{len_m}.{len_n}.{len_k}_'
            '{tile_m}.{tile_n}.{tile_k}_'
            '{pgrid_rows}.{pgrid_cols}_'
            '{blk_rows}.{blk_cols}').format(**locals())
makedirs(job_name, exist_ok=force_generate_job)

# read  templates
cmd_template = '{}/scripts/daint/cmd.sh.templ'.format(source_dir)
with open(cmd_template, 'r') as f:
    cmd_text = f.read().format(**locals())

job_template = '{}/scripts/daint/job.sh.templ'.format(source_dir)
with open(job_template, 'r') as f:
    job_text = f.read().format(**locals())

# write shell scripts
cmd_sh = '{}/cmd.sh'.format(job_name)
with open(cmd_sh, 'w') as f:
    f.write(cmd_text)
chmod(cmd_sh, stat(cmd_sh).st_mode | S_IEXEC)

job_sh = '{}/job.sh'.format(job_name)
with open(job_sh, 'w') as f:
    f.write(job_text)
chmod(job_sh, stat(job_sh).st_mode | S_IEXEC)
