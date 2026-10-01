set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pagerank/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "pagerank / graph500-24 (M) — max_mem_30G workers_16\nmedian=24.493s  mean=24.635s  std=0.249s  min=24.420s  max=24.976s  p90=24.914s  p95=24.945s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:1.72858]
set xrange [23.655410:25.740586]
set grid y
set key top right
set arrow from 24.492824,0 to 24.492824,1.72858 nohead lc rgb 'red' lw 2
set label 'median 24.493s' at 24.492824,1.72858 offset char 0,1 tc rgb 'red'
set arrow from 24.914355,0 to 24.914355,1.72858 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 24.914s' at 24.914355,1.72858 offset char 0,1 tc rgb 'orange'
set arrow from 24.945195,0 to 24.945195,1.72858 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 24.945s' at 24.945195,1.72858 offset char 0,1 tc rgb 'orange'
set arrow from 24.976034,0 to 24.976034,0.0531871 nohead lc rgb '#666666' lw 1
set arrow from 24.492824,0 to 24.492824,0.0531871 nohead lc rgb '#666666' lw 1
set arrow from 24.419961,0 to 24.419961,0.0531871 nohead lc rgb '#666666' lw 1
set arrow from 24.462021,0 to 24.462021,0.0531871 nohead lc rgb '#666666' lw 1
set arrow from 24.821835,0 to 24.821835,0.0531871 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pagerank/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pagerank/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
