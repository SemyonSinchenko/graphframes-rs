set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/ldbd/cdlp/M/graph500-24/icebug_mem_12G_threads_4/wall_time.png'
set title "icebug cdlp / graph500-24 (M) — mem_12G_threads_4\nmedian=217.972s  mean=217.093s  std=2.642s  min=213.292s  max=219.833s  p90=219.417s  p95=219.625s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.169987]
set xrange [205.916428:227.208511]
set grid y
set key top right
set arrow from 217.972226,0 to 217.972226,0.169987 nohead lc rgb 'red' lw 2
set label 'median 217.972s' at 217.972226,0.169987 offset char 0,1 tc rgb 'red'
set arrow from 219.416754,0 to 219.416754,0.169987 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 219.417s' at 219.416754,0.169987 offset char 0,1 tc rgb 'orange'
set arrow from 219.624652,0 to 219.624652,0.169987 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 219.625s' at 219.624652,0.169987 offset char 0,1 tc rgb 'orange'
set arrow from 219.832551,0 to 219.832551,0.00523036 nohead lc rgb '#666666' lw 1
set arrow from 215.576796,0 to 215.576796,0.00523036 nohead lc rgb '#666666' lw 1
set arrow from 213.292388,0 to 213.292388,0.00523036 nohead lc rgb '#666666' lw 1
set arrow from 218.793058,0 to 218.793058,0.00523036 nohead lc rgb '#666666' lw 1
set arrow from 217.972226,0 to 217.972226,0.00523036 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/ldbd/cdlp/M/graph500-24/icebug_mem_12G_threads_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/ldbd/cdlp/M/graph500-24/icebug_mem_12G_threads_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
