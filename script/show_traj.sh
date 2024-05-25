#!/bin/sh

'''exec' python3 "$0" "$@"
' '''
# -*- coding: utf-8 -*-
import re
import sys
# from nglview.scripts.nglview import main
from src.cpmd.script import main
if __name__ == '__main__':
   # re.subは文字列を正規表現で置き換える．この場合はsys.argv[0]を置き換え
   print(sys.argv[0])
   sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
   print(sys.argv[0])
   sys.exit(main())

