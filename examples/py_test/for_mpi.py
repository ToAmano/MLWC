

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
    for i in range(ave): # もしかするとfor文がmpi全てで回っているかも．
        if rank == 0:
            print("now we are in ave loop ... {}  :: {} {}".format(i,ave,res))
            read_traj = []
            for j in range(size):
                read_traj.append(j)
            else:
                read_traj = None
                symbols   = None
                print("now we are in ave loop ... {}  :: {} {} {}/rank".format(i,ave,res,rank))
    return 0

if __name__ == '__main__':
    main()
                
