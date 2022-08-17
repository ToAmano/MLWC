#!/usr/bin/env python3

# pythonからjupyterを呼び出す方法は以下を参考にした．
# https://github.com/nglviewer/nglview/blob/master/nglview/scripts/nglview.py
#

import argparse
import json
import os
import subprocess
import sys

# from .cmd_example import CMD_EXAMPLE

bin_path = os.path.join(sys.prefix, 'bin')

notebook_dict = {
    "cells": [{
        "cell_type":
        "code",
        "execution_count":
        'null',
        "metadata": {
            "collapsed": True
        },
        "outputs": [],
        "source": [
            "import nglview as nv\n", "import pytraj as pt\n", "\n",
            "traj = pt.iterload('test.nc', top='prmtop')\n",
            "view = nv.show_pytraj(traj)\n", "view"
        ]
    }],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython",
            "version": "3.5.1"
        }
    },
    "nbformat":
    4,
    "nbformat_minor":
    0
}

remote_msg = """
Try to use port = {port}
\033[32m In your local machine, run: \033[0m
    {client_cm}
\033[32m NOTE: you might want to replace {hostname} by full hostname with domain name \033[0m
\033[32m Then open your web browser, copy and paste: \033[0m
    http://localhost:{port}/notebooks/{notebook_name}
"""



# remote_portの確認
def get_remote_port(port=None, notebook_path=''):
    import os, socket
    from nglview.scripts.app import NGLViewApp
    port = NGLViewApp().get_port(port=port)

    username = os.getlogin()
    hostname = socket.gethostname()
    client_cm = "ssh -NL localhost:{port}:localhost:{port} {username}@{hostname}".format(
        username=username, hostname=hostname, port=port)
    base_notebook_name = os.path.basename(notebook_path)
    print(
        remote_msg.format(client_cm=client_cm,
                          port=port,
                          hostname=hostname,
                          notebook_name=base_notebook_name))
    return port



def parse_cml_args(cmd, default_jexe):

    description='''
         description='NGLView: An IPython/Jupyter widget to 
         'interactively view molecular structures and trajectories.
    '''

    # 
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # epilog=CMD_EXAMPLE
        )
    # これが本命のファイル名
    # parser.add_argument(
    #     'command',
    #     nargs='?',
    #     help=
    #     'command could be a topology filename (.pdb, .mol2, .parm7, ...) or \n'
    #     'could be a python script (.py), a notebook (.ipynb). '
    #    'If not given, a notebook will be created with only nglview imported')

    parser.add_argument(
         'command',
         nargs='?',
         help=
         'command could be a trajectory filename (currently only .xyz by CP.x and XDATCAR by  VASP) or \n'
        'If not given, a notebook will be created with only nglview imported')
 
        
    # filename
    #parser.add_argument('traj',
    #                    nargs='?',
    #                    help='coordinate filename, optional (depricate)')

    # how to PBC ?
    parser.add_argument(
         '--jump',
         nargs='?',
         default=False,
         help=
         'how to treat periodic boundary condition. If true, atoms stay in the cell, \n'
         'while atoms move across the cell if False. \n'
         'Recommend True for liquid, False for crystal. \n'
         ' Currently only available in .xyz. ')
      
    # 
    #parser.add_argument('-c', '--crd', help='coordinate filename (depricate)')

    # ブラウザ
    parser.add_argument('--browser', help='web browser')

    # 
    parser.add_argument('-j',
                        '--jexe',
                        default=default_jexe,
                        help='jupyter path')
    
    parser.add_argument('--notebook-name',
                        default='tmpnb_ngl.ipynb',
                        help='notebook name')

    parser.add_argument('--port', type=int, help='port number')

    
    parser.add_argument('--remote',
                        action='store_true',
                        help='create remote notebook')

    
    parser.add_argument('--clean',
                        action='store_true',
                        help='delete temp file after closing notebook')

    
    parser.add_argument('--auto',
                        action='store_true',
                        help='Run 1st cell right after openning notebook')

    
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symlink for nglview-js-widgets (developer mode)')
    # args = parser.parse_args(cmd)

    
    return parser.parse_args(cmd)


# main関数
def main(notebook_dict=notebook_dict, cmd=None):
    # typte: (Dict, List[str]) -> None
    pyv_full_string = ','.join(str(i) for i in sys.version_info)
    pyv_short_string = str(sys.version_info[0])
    default_jexe = ' '.join((sys.executable, '-m jupyter_core'))

    args=parse_cml_args(cmd, default_jexe)

    # ファイル名を取得
    command = args.command

    # jump ?
    JUMP = args.jump

    # file absolute path
    original_dirpath=os.path.dirname(__file__)
    
    
    # installなどのコマンドの時
    if command in ['install', 'enable', 'uninstall']:
        cmds = [
            'jupyter', 'nbextension', command, '--py', '--sys-prefix',
            'nglview'
        ]
        if command == 'install':
            cmds.append('--overwrite')
        if args.symlink:
            cmds.append('--symlink')
        subprocess.check_call(cmds)
        sys.exit(0)

    # trajectory file
    # crd = args.traj if args.traj is not None else args.crd

    #if crd is None:
    #    crd = command

    # browserの設定
    browser = '--browser ' + args.browser if args.browser else ''

    create_new_nb = False



    
    if command is not None and command.endswith('.ipynb'):
        notebook_name = command
    else:
        notebook_name = args.notebook_name
        # command=None
        if command is None:
            # create a notebook and just import nglview
            notebook_dict['cells'][0]['source'] = simple_source
            nb_json = json.dumps(notebook_dict)
            nb_json = nb_json.replace('"null"', 'null')
        # これがまさに今求めているやつ
        elif command.endswith('XDATCAR'):
            # open
            first_line='\'1i filename=\"'+str(command)+'\"\''

            #first_line='\'1i filename=\''+str(command)+'\'\''
            print(first_line)
            res=subprocess.run(["pwd"], shell=True)
            print("res",res)
            #print(res.returncode)
            #print(res.output)
            #print(res.stdout)
            #print(res.stderr)

            #dirname = os.path.dirname(os.path.abspath())
            res=subprocess.run(["cp", original_dirpath+"/show_XDATCAR.py", "show_XDATCAR_tmp.py"])

            if not res.returncode==0:
                print("ERROR :: can not copy show_XDATCAR.py")
                sys.exit(1)

            
            # res=subprocess.run(["sed", "-i", "-e", first_line , "show_XDATCAR_tmp.py"], shell=True)
            res=subprocess.run(["sed -i -e "+first_line+" show_XDATCAR_tmp.py"], shell=True)
            
            print(res)
            if not res.returncode==0:
                print("ERROR :: can not make show_XDATCAR_tmp.py")
                sys.exit(1)
            #subprocess.call("cat "+first_line+" show_XDATCAR.py > show_XDATCAR_tmp.py")
            
            pycontent = open("show_XDATCAR_tmp.py").read().strip()
            notebook_dict['cells'][0]['source'] = pycontent
            # python to json
            nb_json = json.dumps(notebook_dict)
            #
            # subprocess.call(["rm show_XDATCAR_tmp.py"], shell=True)


        # xyzの場合．jumpの有無も確認する．
        elif command.endswith('.xyz'):
            # open
            second_line='\'1i JUMP='+str(JUMP)+'\''
            first_line='\'1i filename=\"'+str(command)+'\"\''
            
            #first_line='\'1i filename=\''+str(command)+'\'\''
            # print(first_line)
            # res=subprocess.run(["pwd"], shell=True)
            # print("res",res)
            #print(res.returncode)
            #print(res.output)
            #print(res.stdout)
            #print(res.stderr)

            #dirname = os.path.dirname(os.path.abspath())
            res=subprocess.run(["cp", original_dirpath+"/show_CP.py", "show_CP_tmp.py"])

            if not res.returncode==0:
                print("ERROR :: can not copy show_CP.py")
                sys.exit(1)

            
            # res=subprocess.run(["sed", "-i", "-e", first_line , "show_XDATCAR_tmp.py"], shell=True)
            res=subprocess.run(["sed -i -e "+second_line+" show_CP_tmp.py"], shell=True)
            res=subprocess.run(["sed -i -e "+first_line+" show_CP_tmp.py"], shell=True)
            
            print(res)
            if not res.returncode==0:
                print("ERROR :: can not make show_CP_tmp.py")
                sys.exit(1)
            #subprocess.call("cat "+first_line+" show_XDATCAR.py > show_XDATCAR_tmp.py")
            
            pycontent = open("show_CP_tmp.py").read().strip()
            notebook_dict['cells'][0]['source'] = pycontent
            # python to json
            nb_json = json.dumps(notebook_dict)
            #
            # subprocess.call(["rm show_XDATCAR_tmp.py"], shell=True)

        else:
            nb_json = json.dumps(notebook_dict)
            # original including crd
            #nb_json = nb_json.replace('"null"', 'null').replace(
            #    'test.nc', crd).replace('prmtop', command)
            nb_json = nb_json.replace('"null"', 'null').replace('prmtop', command)

            assert os.path.exists(command), '{} does not exists'.format(
                command)

        nb_json = nb_json.replace('"null"', 'null')

        # making ipynb file
        with open(notebook_name, 'w') as fh:
            fh.write(nb_json)
            create_new_nb = True

    dirname = os.path.dirname(os.path.abspath(notebook_name))
    #
    if not args.remote:
        cm = '{jupyter} notebook {notebook_name} {browser}'.format(
            jupyter=args.jexe, notebook_name=notebook_name, browser=browser)
    else:
        port = get_remote_port(args.port, notebook_name)
        cm = '{jupyter} notebook --no-browser --port {port} ' \
              '--notebook-dir {dirname}'.format(jupyter=args.jexe,
                                                port=port,
                                                dirname=dirname)
        print('NOTE: make sure to open {} in your local machine\n'.format(
            notebook_name))
    
    try:
        subprocess.check_call(cm.split())
    except KeyboardInterrupt:
        if args.clean and create_new_nb:
            print(f"deleting {notebook_name}")
            os.remove(notebook_name)
        if args.auto:
            disable_extension(jupyter=args.jexe)


if __name__ == '__main__':
    main(cmd=sys.argv[1:])
