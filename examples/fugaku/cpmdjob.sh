#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "elapse=0:50:00"
#PJM -g "hp220331"
#PJM --name "cpmd_job"
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

# **********************
# 
# #PJM -L  "node=2x1x2:torus" 使用するノードの数を指定
# #PJM --mpi "proc=16" mpi並列数を指定．この数を使用するノード数でわると1ノードでのmpi数になる．
# #PJM --mpi "max-proc-per-node=4" 1ノードでのmpi数を指定．上の"proc"とconsistentになるように設定する．
# openMPの設定は，プリアンブルでは全く必要ない．これはexport PARALLEL以下の3行で指定する．
# **********************

# load modules
. /vol0004/apps/oss/spack/share/spack/setup-env.sh

spack load fujitsu-fftw #@1.1.0%fj@4.8.0 arch=linux-rhel8-a64fx
spack load fujitsu-mpi  #@head%fj@4.8.0 arch=linux-rhel8-a64fx
spack load libxc@4.3.4  #%fj@4.8.0 arch=linux-rhel8-a64fx

# execute
echo "start mpiexec with -stdout-proc..."
mpiexec -stdout-proc bomd-nonwan.out /vol0006/mdt1/data/hp220331/src/CPMD-4.3/bin/cpmd_FUGAKU-MPI-FFTW/bin/cpmd.x bomd-nonwan.inp

