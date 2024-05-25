

def test(input):
    if input == None:
        return 0
    else:
        return input+1

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
    res = 3

    if res != 0:
        if rank == 0:
            print(" size :: {}".format(size))
            print("now we are in final step... :: {} {}".format(ave,res))
            read_traj = []
            for j in range(res):                
                read_traj.append(j)
            for i in range(size - res):
                read_traj.append(None) # ひょっとするとここがNoneだと計算が回らない？
            if len(read_traj) != size:
                print("")
                print("ERROR :: len(read_traj) != size")
                print("")
            print("len(read_traj) :: {}".format(len(read_traj)))
            print("read_traj :: {}".format(read_traj))
            result_dipole_tmp = None # あとで使うので
        else: # rank != 0
            read_traj = None
            result_dipole_tmp = None # あとで使うので
        #
        # bcast/scatter data
        read_traj = comm.scatter(read_traj,root=0)
        if read_traj == None: # sacatterした後にNoneのままだったら，計算しない．
            aseatom = None
        else:
            aseatom   = read_traj
        fr = ave*size+rank
        print(" fr is ... {}/aseatom :: {}/rank {}/size".format(aseatom,rank,size))
        # print(" hello rank {} {}".format(rank, read_traj))
        if rank == 0:
            print("")
            print(" finish scattering data ...")
            print("")
        # frに変数が必要
        result_dipole_tmp = test(aseatom)
        print(" result_dipole_tmp {}/rank {}".format(rank, result_dipole_tmp))
        result_dipole_tmp = comm.gather(result_dipole_tmp, root=0) 
        if rank == 0:
            print("")
            print(" finish descripter calculation ...")
            print(" result_dipole_tmp {}/rank {}".format(rank, result_dipole_tmp))
        # else:
        #    print(" result_dipole_tmp {}/rank {}".format(rank, result_dipole_tmp))
    return 0


if __name__ == '__main__':
    main()
                
