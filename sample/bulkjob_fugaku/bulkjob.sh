#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "elapse=0:50:00"
#PJM -g "hp220331"
#PJM --name "pg_4n_12t"
#PJM -L "freq=2200,eco_state=2"
#PJM --mpi "proc=16"
#PJM -L  "node=2x1x2:torus"
#PJM --mpi "max-proc-per-node=4"
#PJM --mpi "rank-map-bychip"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --llio "localtmp-size=10Gi"
#PJM --mail-list tragic44cg@icloud.com
#PJM -m b
#PJM -m e
#PJM -s
export PARALLEL=12
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=${PARALLEL}

# pythonが競合するのでsystemで実行する必要がある．
pyenv global system
. /vol0004/apps/oss/spack/share/spack/setup-env.sh

spack load fujitsu-fftw #@1.1.0%fj@4.8.0 arch=linux-rhel8-a64fx
spack load fujitsu-mpi  #@head%fj@4.8.0 arch=linux-rhel8-a64fx
spack load libxc@4.3.4  #%fj@4.8.0 arch=linux-rhel8-a64fx

# execute
# execute 
cd bulkjob/struc_${PJM_BULKNUM} #バルク番号に展開

echo "start mpiexec with -stdout-proc..."
mpiexec -stdout-proc bomd-oneshot.out /vol0006/mdt1/data/hp220331/src/CPMD-4.3/bin/cpmd_FUGAKU-MPI-FFTW/bin/cpmd.x bomd-oneshot.inp
# 終了後restart.1を削除する．
rm tmp/RESTART.1

cd ../../

# pythonが競合するのでsystemで実行する必要がある．
# pyenv global anaconda3-2022.05


