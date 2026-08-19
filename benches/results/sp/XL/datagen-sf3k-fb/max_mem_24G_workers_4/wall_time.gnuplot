set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/sp/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.png'
set title "sp / datagen-sf3k-fb (XL) — max_mem_24G workers_4\nmedian=1146.889s  mean=1162.893s  std=28.978s  min=1137.517s  max=1207.340s  p90=1195.136s  p95=1201.238s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0182097]
set xrange [1066.567338:1278.289350]
set grid y
set key top right
set arrow from 1146.888607,0 to 1146.888607,0.0182097 nohead lc rgb 'red' lw 2
set label 'median 1146.889s' at 1146.888607,0.0182097 offset char 0,1 tc rgb 'red'
set arrow from 1195.135516,0 to 1195.135516,0.0182097 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1195.136s' at 1195.135516,0.0182097 offset char 0,1 tc rgb 'orange'
set arrow from 1201.237615,0 to 1201.237615,0.0182097 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1201.238s' at 1201.237615,0.0182097 offset char 0,1 tc rgb 'orange'
set arrow from 1207.339715,0 to 1207.339715,0.000560297 nohead lc rgb '#666666' lw 1
set arrow from 1145.891866,0 to 1145.891866,0.000560297 nohead lc rgb '#666666' lw 1
set arrow from 1146.888607,0 to 1146.888607,0.000560297 nohead lc rgb '#666666' lw 1
set arrow from 1176.829216,0 to 1176.829216,0.000560297 nohead lc rgb '#666666' lw 1
set arrow from 1137.516973,0 to 1137.516973,0.000560297 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/sp/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/sp/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
