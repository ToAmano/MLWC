#
# simple code to make gif of nglview
#
# http://nglviewer.org/nglview/latest/_modules/nglview/contrib/movie.html


# make sure to change your web browser option to save files as default (vs open file by external program)
# NGLView will render each snapshot and save image to your web browser default download location

from time import sleep
import numpy as np
import os
import shutil
import moviepy.editor as mpy
from PIL import Image

# https://ambermd.org/tutorials/analysis/tutorial_notebooks/nglview_movie/index.html


# current_dir=os.getcwd()
# os.mkdir(current_dir+"/tmp")

# save frames every 10 step
for frame in np.arange(0,len(traj),10):
    # set frame to update coordinates
    view.frame = int(frame)
    # make sure to let NGL spending enough time to update coordinates
    sleep(0.2)
    view.download_image(filename="0image{}.png".format(frame))
    # make sure to let NGL spending enough time to render before going to next frame
    sleep(0.2)



# In my case, my default download folder is /Users/amano/var/chrome/0image{}.png
# caution !! change here !!
dwn_dir = os.environ['HOME']+"/Downloads/"
template_dir =dwn_dir+"tmp_fig/"

os.mkdir(template_dir)

for i in np.arange(0,len(traj),10):
    shutil.move(dwn_dir+"0image{}.png".format(str(i)), template_dir)

# In my case, my default download folder is /Users/amano/var/chrome/0image{}.png
# caution !! change here !!
template=template_dir+"0image{}.png"


# get all (sorted) image files
imagefiles = [template.format(str(i)) for i in np.arange(0,len(traj),10)]

pictures=[]

for i in np.arange(0,len(traj),10):
    img=Image.open(template.format(str(i)))
    pictures.append(img)
    
#gifアニメを出力する
pictures[0].save(filename+"_animation.gif",save_all=True, append_images=pictures[1:],\
optimize=False, duration=125, loop=0)


# 最後に中間ファイルを削除する．
shutil.rmtree(template_dir)

#durationは１枚の画像を表示する期間をミリ秒単位で表しています。
#つまりduration=500で４枚の画像を合体させると500×4=2000ミリ秒⇒2秒の動画を作成することになります。
# frame_per_secに直すには，frame_per_sec*duration=1000 に注意．    
# fps=8ならduration=125
    
# make a gif file

#frame_per_second = 8
#im = mpy.ImageSequenceClip(imagefiles, fps=frame_per_second)
#print(im)

# im.save(‘/Users/amano/var/chrome/mytest.gif’
#              save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

#im.write_gif(‘my_movie.gif’, fps=frame_per_second)
#im.close()


# nglviewのを使った場合

# http://nglviewer.org/nglview/latest/_api/nglview.contrib.movie.html
# from nglview.contrib.movie import MovieMaker
# download_folder = "/Users/amano/var/chrome"
# output=output,
# download_folder=download_folder,
# mov = MovieMaker(view2,   step=100 , output="my_test.gif", timeout=0.2, in_memory=False)
#mov = MovieMaker(view2,   step=10 ,output=‘my.avi’, in_memory=True, moviepy_params=moviepy_params)
# mov.make()
