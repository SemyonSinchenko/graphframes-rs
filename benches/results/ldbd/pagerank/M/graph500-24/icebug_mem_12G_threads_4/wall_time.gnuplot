set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/ldbd/pagerank/M/graph500-24/icebug_mem_12G_threads_4/wall_time.png'
set title "icebug pagerank / graph500-24 (M) — mem_12G_threads_4\nmedian=209.151s  mean=208.871s  std=1.421s  min=206.508s  max=210.261s  p90=209.982s  p95=210.121s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.627813]
set xrange [204.930716:211.838179]
set grid y
set key top right
set arrow from 209.150818,0 to 209.150818,0.627813 nohead lc rgb 'red' lw 2
set label 'median 209.151s' at 209.150818,0.627813 offset char 0,1 tc rgb 'red'
set arrow from 209.981513,0 to 209.981513,0.627813 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 209.982s' at 209.981513,0.627813 offset char 0,1 tc rgb 'orange'
set arrow from 210.121246,0 to 210.121246,0.627813 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 210.121s' at 210.121246,0.627813 offset char 0,1 tc rgb 'orange'
set arrow from 209.562314,0 to 209.562314,0.0193173 nohead lc rgb '#666666' lw 1
set arrow from 210.260979,0 to 210.260979,0.0193173 nohead lc rgb '#666666' lw 1
set arrow from 206.507916,0 to 206.507916,0.0193173 nohead lc rgb '#666666' lw 1
set arrow from 208.874581,0 to 208.874581,0.0193173 nohead lc rgb '#666666' lw 1
set arrow from 209.150818,0 to 209.150818,0.0193173 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/ldbd/pagerank/M/graph500-24/icebug_mem_12G_threads_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/ldbd/pagerank/M/graph500-24/icebug_mem_12G_threads_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
