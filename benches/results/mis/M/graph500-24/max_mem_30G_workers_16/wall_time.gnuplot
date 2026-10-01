set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/mis/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "mis / graph500-24 (M) — max_mem_30G workers_16\nmedian=113.434s  mean=112.867s  std=2.733s  min=108.843s  max=116.249s  p90=115.303s  p95=115.776s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.219339]
set xrange [104.354147:120.738580]
set grid y
set key top right
set arrow from 113.433913,0 to 113.433913,0.219339 nohead lc rgb 'red' lw 2
set label 'median 113.434s' at 113.433913,0.219339 offset char 0,1 tc rgb 'red'
set arrow from 115.302609,0 to 115.302609,0.219339 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 115.303s' at 115.302609,0.219339 offset char 0,1 tc rgb 'orange'
set arrow from 115.776028,0 to 115.776028,0.219339 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 115.776s' at 115.776028,0.219339 offset char 0,1 tc rgb 'orange'
set arrow from 113.433913,0 to 113.433913,0.00674889 nohead lc rgb '#666666' lw 1
set arrow from 108.843281,0 to 108.843281,0.00674889 nohead lc rgb '#666666' lw 1
set arrow from 113.882354,0 to 113.882354,0.00674889 nohead lc rgb '#666666' lw 1
set arrow from 111.924882,0 to 111.924882,0.00674889 nohead lc rgb '#666666' lw 1
set arrow from 116.249446,0 to 116.249446,0.00674889 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/mis/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/mis/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
