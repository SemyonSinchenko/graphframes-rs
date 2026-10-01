set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/hyperanf/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "hyperanf / cit-Patents (XS) — max_mem_30G workers_16\nmedian=70.711s  mean=70.970s  std=1.010s  min=69.985s  max=72.667s  p90=71.969s  p95=72.318s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:1.18654]
set xrange [69.158149:73.493873]
set grid y
set key top right
set arrow from 70.711064,0 to 70.711064,1.18654 nohead lc rgb 'red' lw 2
set label 'median 70.711s' at 70.711064,1.18654 offset char 0,1 tc rgb 'red'
set arrow from 71.969130,0 to 71.969130,1.18654 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 71.969s' at 71.969130,1.18654 offset char 0,1 tc rgb 'orange'
set arrow from 72.317831,0 to 72.317831,1.18654 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 72.318s' at 72.317831,1.18654 offset char 0,1 tc rgb 'orange'
set arrow from 69.985491,0 to 69.985491,0.036509 nohead lc rgb '#666666' lw 1
set arrow from 70.711064,0 to 70.711064,0.036509 nohead lc rgb '#666666' lw 1
set arrow from 70.562266,0 to 70.562266,0.036509 nohead lc rgb '#666666' lw 1
set arrow from 70.923026,0 to 70.923026,0.036509 nohead lc rgb '#666666' lw 1
set arrow from 72.666532,0 to 72.666532,0.036509 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/hyperanf/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/hyperanf/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
