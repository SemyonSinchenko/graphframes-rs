set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pagerank/2XL/graph500-28/max_mem_30G_workers_16/wall_time.png'
set title "pagerank / graph500-28 (2XL) — max_mem_30G workers_16\nmedian=911.849s  mean=911.418s  std=2.609s  min=908.103s  max=914.425s  p90=913.945s  p95=914.185s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.151599]
set xrange [900.085927:922.441748]
set grid y
set key top right
set arrow from 911.848879,0 to 911.848879,0.151599 nohead lc rgb 'red' lw 2
set label 'median 911.849s' at 911.848879,0.151599 offset char 0,1 tc rgb 'red'
set arrow from 913.945467,0 to 913.945467,0.151599 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 913.945s' at 913.945467,0.151599 offset char 0,1 tc rgb 'orange'
set arrow from 914.185130,0 to 914.185130,0.151599 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 914.185s' at 914.185130,0.151599 offset char 0,1 tc rgb 'orange'
set arrow from 913.226478,0 to 913.226478,0.00466458 nohead lc rgb '#666666' lw 1
set arrow from 908.102882,0 to 908.102882,0.00466458 nohead lc rgb '#666666' lw 1
set arrow from 914.424793,0 to 914.424793,0.00466458 nohead lc rgb '#666666' lw 1
set arrow from 911.848879,0 to 911.848879,0.00466458 nohead lc rgb '#666666' lw 1
set arrow from 909.485156,0 to 909.485156,0.00466458 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pagerank/2XL/graph500-28/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pagerank/2XL/graph500-28/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
