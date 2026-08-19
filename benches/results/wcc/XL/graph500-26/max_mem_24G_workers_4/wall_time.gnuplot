set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/graph500-26/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-26 (XL) — max_mem_24G workers_4\nmedian=1517.643s  mean=1520.475s  std=40.188s  min=1481.146s  max=1574.198s  p90=1562.895s  p95=1568.547s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.00988275]
set xrange [1357.644576:1697.699252]
set grid y
set key top right
set arrow from 1517.642558,0 to 1517.642558,0.00988275 nohead lc rgb 'red' lw 2
set label 'median 1517.643s' at 1517.642558,0.00988275 offset char 0,1 tc rgb 'red'
set arrow from 1562.895300,0 to 1562.895300,0.00988275 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1562.895s' at 1562.895300,0.00988275 offset char 0,1 tc rgb 'orange'
set arrow from 1568.546771,0 to 1568.546771,0.00988275 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1568.547s' at 1568.546771,0.00988275 offset char 0,1 tc rgb 'orange'
set arrow from 1517.642558,0 to 1517.642558,0.000304085 nohead lc rgb '#666666' lw 1
set arrow from 1574.198241,0 to 1574.198241,0.000304085 nohead lc rgb '#666666' lw 1
set arrow from 1545.940889,0 to 1545.940889,0.000304085 nohead lc rgb '#666666' lw 1
set arrow from 1481.145587,0 to 1481.145587,0.000304085 nohead lc rgb '#666666' lw 1
set arrow from 1483.450075,0 to 1483.450075,0.000304085 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
