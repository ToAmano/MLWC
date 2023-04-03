#!/bin/bash
#SBATCH -p mem2      # キューの指定
#SBATCH -n 56         # CPU数の指定
# #SBATCH --mem 5020G  # メモリ量の指定
#SBATCH -t 24:00:00           # ジョブ実行時間の制限を指定
# #PJM -L "node=2"
# #PJM -L "rscgrp=small"
# #PJM -L "elapse=72:00:00"
#PJM -g "hp220331"
#PJM --name "prepost_test"
# #PJM --mpi "max-proc-per-node=48"
#PJM -x PJM_LLIO_GFSCACHE=/vol0004
#PJM --llio "localtmp-size=10Gi"
#PJM --mail-list tragic44cg@icloud.com
#PJM -m b
#PJM -m e
#PJM -s
export PLE_MPI_STD_EMPTYFILE=off
export OMP_NUM_THREADS=1

# pythonが競合するのでsystemで実行する必要がある．
pyenv global system
. /vol0004/apps/oss/spack/share/spack/setup-env.sh

spack load fujitsu-fftw@1.1.0%fj@4.8.0 arch=linux-rhel8-a64fx
spack load fujitsu-mpi@head%fj@4.8.0 arch=linux-rhel8-a64fx
spack load libxc@4.3.4%fj@4.8.0 arch=linux-rhel8-a64fx

spack load quantum-espresso@6.8
# spack load gromacs@2022.3%fj@4.8.0

# pythonが競合するのでsystemで実行する必要がある．
pyenv global anaconda3-2022.05

# execute 
# cd gromacs_test
# python gromacs.py
# cd ..


# dir=`dirname $0`
dir=$SLURM_SUBMIT_DIR
cd $dir/gromacs_test

echo " 現在のディレクトリ"
pwd

#grompp
/vol0006/mdt1/data/hp220331/src/gromacs-2022.4/build/bin/gmx grompp -f em.mdp -p system.top -c init.gro -o em.tpr -maxwarn 10
# コマンドの終了ステータスが「0」以外であれば、コマンドは成功していない
cmdstatus=$?
if [ $cmdstatus -ne 0 ]; then
    echo "command error :: gmx grompp !!"
    # 必要なら、ここで異常系フローの後処理ができる

    # 実行を終了させる
    exit $cmdstatus
fi


#mdrun for Equilibration
/vol0006/mdt1/data/hp220331/src/gromacs-2022.4/build/bin/gmx mdrun -s em.tpr -o em.trr -e em.edr -c em.gro -nb cp
# コマンドの終了ステータスが「0」以外であれば、コマンドは成功していない
cmdstatus=$?
if [ $cmdstatus -ne 0 ]; then
    echo "command error :: gmx mdrun !!"
    # 必要なら、ここで異常系フローの後処理ができる

    # 実行を終了させる
    exit $cmdstatus
fi


#grompp (入力ファイルを作成)
/vol0006/mdt1/data/hp220331/src/gromacs-2022.4/build/bin/gmx grompp -f run.mdp -p system.top -c em.gro -o eq.tpr -maxwarn 1
# コマンドの終了ステータスが「0」以外であれば、コマンドは成功していない
cmdstatus=$?
if [ $cmdstatus -ne 0 ]; then
    echo "command error :: gmx grompp !!"
    # 必要なら、ここで異常系フローの後処理ができる

    # 実行を終了させる
    exit $cmdstatus
fi


#mdrun (eq.groを作成)
/vol0006/mdt1/data/hp220331/src/gromacs-2022.4/build/bin/gmx mdrun -s eq.tpr -o eq.trr -e eq.edr -c eq.gro -nb cpu
# コマンドの終了ステータスが「0」以外であれば、コマンドは成功していない
cmdstatus=$?
if [ $cmdstatus -ne 0 ]; then
    echo "command error :: gmx mdrun !!"
    # 必要なら、ここで異常系フローの後処理ができる

    # 実行を終了させる
    exit $cmdstatus
fi




# 最後にeq_pbc.trrを作成する．
echo "System" > ./inputs/anal.txt
/vol0006/mdt1/data/hp220331/src/gromacs-2022.4/build_mpi/bin/gmx_mpi trjconv -s eq.tpr -f eq.trr -dump 0 -o eq.pdb < ./inputs/anal.txt
/vol0006/mdt1/data/hp220331/src/gromacs-2022.4/build_mpi/bin/gmx_mpi trjconv -s eq.tpr -f eq.trr -pbc mol -force -o eq_pbc.trr < ./inputs/anal.txt

cd $dir
