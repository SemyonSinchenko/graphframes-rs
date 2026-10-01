set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/sp/2XL/graph500-28/max_mem_30G_workers_16/wall_time.png'
set title "sp / graph500-28 (2XL) — max_mem_30G workers_16\nmedian=782.627s  mean=782.937s  std=0.528s  min=782.506s  max=783.717s  p90=783.530s  p95=783.624s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.874607]
set xrange [780.974415:785.248963]
set grid y
set key top right
set arrow from 782.627228,0 to 782.627228,0.874607 nohead lc rgb 'red' lw 2
set label 'median 782.627s' at 782.627228,0.874607 offset char 0,1 tc rgb 'red'
set arrow from 783.530177,0 to 783.530177,0.874607 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 783.530s' at 783.530177,0.874607 offset char 0,1 tc rgb 'orange'
set arrow from 783.623572,0 to 783.623572,0.874607 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 783.624s' at 783.623572,0.874607 offset char 0,1 tc rgb 'orange'
set arrow from 783.249989,0 to 783.249989,0.026911 nohead lc rgb '#666666' lw 1
set arrow from 782.506410,0 to 782.506410,0.026911 nohead lc rgb '#666666' lw 1
set arrow from 783.716968,0 to 783.716968,0.026911 nohead lc rgb '#666666' lw 1
set arrow from 782.627228,0 to 782.627228,0.026911 nohead lc rgb '#666666' lw 1
set arrow from 782.581968,0 to 782.581968,0.026911 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/sp/2XL/graph500-28/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/sp/2XL/graph500-28/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
