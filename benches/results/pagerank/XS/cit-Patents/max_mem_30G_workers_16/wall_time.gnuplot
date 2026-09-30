set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pagerank/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "pagerank / cit-Patents (XS) — max_mem_30G workers_16\nmedian=4.046s  mean=4.052s  std=0.029s  min=4.016s  max=4.096s  p90=4.081s  p95=4.089s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:26.2646]
set xrange [3.979579:4.132071]
set grid y
set key top right
set arrow from 4.045853,0 to 4.045853,26.2646 nohead lc rgb 'red' lw 2
set label 'median 4.046s' at 4.045853,26.2646 offset char 0,1 tc rgb 'red'
set arrow from 4.081407,0 to 4.081407,26.2646 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 4.081s' at 4.081407,26.2646 offset char 0,1 tc rgb 'orange'
set arrow from 4.088514,0 to 4.088514,26.2646 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 4.089s' at 4.088514,26.2646 offset char 0,1 tc rgb 'orange'
set arrow from 4.045853,0 to 4.045853,0.808141 nohead lc rgb '#666666' lw 1
set arrow from 4.016030,0 to 4.016030,0.808141 nohead lc rgb '#666666' lw 1
set arrow from 4.044194,0 to 4.044194,0.808141 nohead lc rgb '#666666' lw 1
set arrow from 4.060089,0 to 4.060089,0.808141 nohead lc rgb '#666666' lw 1
set arrow from 4.095620,0 to 4.095620,0.808141 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pagerank/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pagerank/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
