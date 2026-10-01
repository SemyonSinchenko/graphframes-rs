set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/wcc/L/graph500-25/max_mem_30G_workers_16/wall_time.png'
set title "wcc / graph500-25 (L) — max_mem_30G workers_16\nmedian=82.458s  mean=81.918s  std=1.090s  min=80.194s  max=82.893s  p90=82.748s  p95=82.820s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.540143]
set xrange [77.862920:85.223839]
set grid y
set key top right
set arrow from 82.457698,0 to 82.457698,0.540143 nohead lc rgb 'red' lw 2
set label 'median 82.458s' at 82.457698,0.540143 offset char 0,1 tc rgb 'red'
set arrow from 82.747559,0 to 82.747559,0.540143 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 82.748s' at 82.747559,0.540143 offset char 0,1 tc rgb 'orange'
set arrow from 82.820156,0 to 82.820156,0.540143 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 82.820s' at 82.820156,0.540143 offset char 0,1 tc rgb 'orange'
set arrow from 82.457698,0 to 82.457698,0.0166198 nohead lc rgb '#666666' lw 1
set arrow from 82.529770,0 to 82.529770,0.0166198 nohead lc rgb '#666666' lw 1
set arrow from 81.513307,0 to 81.513307,0.0166198 nohead lc rgb '#666666' lw 1
set arrow from 82.892752,0 to 82.892752,0.0166198 nohead lc rgb '#666666' lw 1
set arrow from 80.194007,0 to 80.194007,0.0166198 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/wcc/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/wcc/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
