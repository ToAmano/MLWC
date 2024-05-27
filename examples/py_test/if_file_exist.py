
def main():
    import os
    import sys

    print(" ==DEBUG== start main() !!")

    # * ここからMPI implementation
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # itpファイルの読み込み
    if rank == 0:
        # * 1-1：コマンドライン引数の読み込み
        inputfilename=sys.argv[1]
        # include.small.if_file_exist(inputfilename) # ファイルの存在確認（どうもmpiだとうまく動かない？）
        is_file = os.path.isfile(inputfilename)
        if not is_file: # itpファイルの存在を確認
            print("ERROR not found the file :: {} !! ".format(inputfilename))
            sys.exit("1")
        else:
            print("input file found !!")
        return 0
