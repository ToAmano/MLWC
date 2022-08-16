# https://ss.scphys.kyoto-u.ac.jp/person/yonezawa/contents/program/gnuplot/paper_adv2.html
# Time-stamp: <2022-07-22 11:41:06 amano>


# output setting
set term pdfcairo enhanced # enhanced for latin
set output "test7.pdf"


# if y2 axis is used ?
# http://blog.livedoor.jp/mwalker/archives/27403552.html
# set y2tics

# title
#set title "A2u"

# axis label
# set xlabel "Displacement of O [Ang]"
# set ylabel "Energy [eV]"
# set y2label "y2"

# margin setting (if num is large, margin becomes large)
set lmargin 15 #left
set bmargin 5 #bottom


# set xrange [0:0.2]
set key left top

# plot
plot \
  "output.txt" u 2:3 title "harm",\


    gnuplot
    plot "outdir/si.evp" using 2:5 with linespoints

plot total energy
    gnuplot
    plot "outdir/si.evp" using 2:6 with linespoints