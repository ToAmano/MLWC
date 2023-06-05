

def main():
    import os
    import sys

    print(" ==DEBUG== start main() !!")

    # * ここからMPI implementation
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    ave = 100
    res = 20
    if rank == 0:
        read_traj = []
        for j in range(3):
            read_traj.append(j)
        print("len(read_traj) {}".format(len(read_traj)))  
    else:
        read_traj = None
        symbols   = None
    read_traj = comm.scatter(read_traj,root=0)
    result_dipole_tmp = read_traj+1
    result_dipole_tmp = comm.gather(result_dipole_tmp,root=0)
    if rank == 0:
        print(result_dipole_tmp)
    return 0

if __name__ == '__main__':
    main()
                
