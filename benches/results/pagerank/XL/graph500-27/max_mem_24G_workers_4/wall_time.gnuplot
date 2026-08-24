set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/XL/graph500-27/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / graph500-27 (XL) — max_mem_24G workers_4\nmedian=1076.382s  mean=1079.427s  std=9.832s  min=1070.777s  max=1096.120s  p90=1089.397s  p95=1092.759s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0948077]
set xrange [1059.833074:1107.063837]
set grid y
set key top right
set arrow from 1076.382133,0 to 1076.382133,0.0948077 nohead lc rgb 'red' lw 2
set label 'median 1076.382s' at 1076.382133,0.0948077 offset char 0,1 tc rgb 'red'
set arrow from 1089.397281,0 to 1089.397281,0.0948077 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1089.397s' at 1089.397281,0.0948077 offset char 0,1 tc rgb 'orange'
set arrow from 1092.758638,0 to 1092.758638,0.0948077 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1092.759s' at 1092.758638,0.0948077 offset char 0,1 tc rgb 'orange'
set arrow from 1076.382133,0 to 1076.382133,0.00291716 nohead lc rgb '#666666' lw 1
set arrow from 1079.313210,0 to 1079.313210,0.00291716 nohead lc rgb '#666666' lw 1
set arrow from 1070.776916,0 to 1070.776916,0.00291716 nohead lc rgb '#666666' lw 1
set arrow from 1096.119995,0 to 1096.119995,0.00291716 nohead lc rgb '#666666' lw 1
set arrow from 1074.541185,0 to 1074.541185,0.00291716 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/XL/graph500-27/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/XL/graph500-27/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
