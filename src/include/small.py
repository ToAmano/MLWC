

# 簡単なパーツ部分のコード


def if_file_exist(filename):
    import os
    import sys
    is_file = os.path.isfile(filename)
    if not is_file: # itpファイルの存在を確認
        print("ERROR not found the file :: {} !! ".format(filename))    
        sys.exit("1")
    return 0
