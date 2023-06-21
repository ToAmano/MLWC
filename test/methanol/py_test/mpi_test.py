

def main():
    # * ここからMPI implementation
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()  
    rank = comm.Get_rank()

    import time
    
    if rank == 0:
        print("test this is rank 0")
        time.sleep(10)
        print("finish sleep")
        test = 10
    else:
        print("this is others")
        test = None
    test = comm.bcast(test,root=0)
    return 0

if __name__ == '__main__':
    main()

