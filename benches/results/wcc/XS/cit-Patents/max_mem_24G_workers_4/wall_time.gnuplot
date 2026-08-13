set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XS/cit-Patents/max_mem_24G_workers_4/wall_time.png'
set title "wcc / cit-Patents (XS) — max_mem_24G workers_4\nmedian=26.855s  mean=26.799s  std=0.143s  min=26.557s  max=26.900s  p90=26.899s  p95=26.899s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:5.31611]
set xrange [26.307493:27.149146]
set grid y
set key top right
set arrow from 26.854517,0 to 26.854517,5.31611 nohead lc rgb 'red' lw 2
set label 'median 26.855s' at 26.854517,5.31611 offset char 0,1 tc rgb 'red'
set arrow from 26.898635,0 to 26.898635,5.31611 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 26.899s' at 26.898635,5.31611 offset char 0,1 tc rgb 'orange'
set arrow from 26.899146,0 to 26.899146,5.31611 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 26.899s' at 26.899146,5.31611 offset char 0,1 tc rgb 'orange'
set arrow from 26.854517,0 to 26.854517,0.163573 nohead lc rgb '#666666' lw 1
set arrow from 26.899658,0 to 26.899658,0.163573 nohead lc rgb '#666666' lw 1
set arrow from 26.788311,0 to 26.788311,0.163573 nohead lc rgb '#666666' lw 1
set arrow from 26.897099,0 to 26.897099,0.163573 nohead lc rgb '#666666' lw 1
set arrow from 26.556980,0 to 26.556980,0.163573 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
