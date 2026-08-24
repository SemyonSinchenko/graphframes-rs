set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/M/graph500-24/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-24 (M) — max_mem_24G workers_4\nmedian=147.666s  mean=147.880s  std=2.536s  min=144.899s  max=151.887s  p90=150.301s  p95=151.094s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.475874]
set xrange [142.850574:153.935364]
set grid y
set key top right
set arrow from 147.666147,0 to 147.666147,0.475874 nohead lc rgb 'red' lw 2
set label 'median 147.666s' at 147.666147,0.475874 offset char 0,1 tc rgb 'red'
set arrow from 150.300677,0 to 150.300677,0.475874 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 150.301s' at 150.300677,0.475874 offset char 0,1 tc rgb 'orange'
set arrow from 151.093957,0 to 151.093957,0.475874 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 151.094s' at 151.093957,0.475874 offset char 0,1 tc rgb 'orange'
set arrow from 147.027760,0 to 147.027760,0.0146423 nohead lc rgb '#666666' lw 1
set arrow from 144.898703,0 to 144.898703,0.0146423 nohead lc rgb '#666666' lw 1
set arrow from 147.920839,0 to 147.920839,0.0146423 nohead lc rgb '#666666' lw 1
set arrow from 147.666147,0 to 147.666147,0.0146423 nohead lc rgb '#666666' lw 1
set arrow from 151.887236,0 to 151.887236,0.0146423 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
