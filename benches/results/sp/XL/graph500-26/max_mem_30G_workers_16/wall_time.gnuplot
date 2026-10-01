set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/sp/XL/graph500-26/max_mem_30G_workers_16/wall_time.png'
set title "sp / graph500-26 (XL) — max_mem_30G workers_16\nmedian=72.862s  mean=72.927s  std=0.457s  min=72.516s  max=73.701s  p90=73.371s  p95=73.536s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:2.16712]
set xrange [72.073703:74.142573]
set grid y
set key top right
set arrow from 72.862183,0 to 72.862183,2.16712 nohead lc rgb 'red' lw 2
set label 'median 72.862s' at 72.862183,2.16712 offset char 0,1 tc rgb 'red'
set arrow from 73.370671,0 to 73.370671,2.16712 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 73.371s' at 73.370671,2.16712 offset char 0,1 tc rgb 'orange'
set arrow from 73.535721,0 to 73.535721,2.16712 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 73.536s' at 73.535721,2.16712 offset char 0,1 tc rgb 'orange'
set arrow from 72.862183,0 to 72.862183,0.0666806 nohead lc rgb '#666666' lw 1
set arrow from 73.700770,0 to 73.700770,0.0666806 nohead lc rgb '#666666' lw 1
set arrow from 72.875523,0 to 72.875523,0.0666806 nohead lc rgb '#666666' lw 1
set arrow from 72.682876,0 to 72.682876,0.0666806 nohead lc rgb '#666666' lw 1
set arrow from 72.515506,0 to 72.515506,0.0666806 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/sp/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/sp/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
