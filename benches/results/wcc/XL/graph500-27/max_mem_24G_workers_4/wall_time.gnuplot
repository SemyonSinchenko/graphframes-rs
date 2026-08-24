set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/graph500-27/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-27 (XL) — max_mem_24G workers_4\nmedian=1587.515s  mean=1591.145s  std=16.735s  min=1575.244s  max=1609.318s  p90=1608.789s  p95=1609.053s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0234183]
set xrange [1523.816456:1660.745693]
set grid y
set key top right
set arrow from 1587.515193,0 to 1587.515193,0.0234183 nohead lc rgb 'red' lw 2
set label 'median 1587.515s' at 1587.515193,0.0234183 offset char 0,1 tc rgb 'red'
set arrow from 1608.788795,0 to 1608.788795,0.0234183 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1608.789s' at 1608.788795,0.0234183 offset char 0,1 tc rgb 'orange'
set arrow from 1609.053435,0 to 1609.053435,0.0234183 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1609.053s' at 1609.053435,0.0234183 offset char 0,1 tc rgb 'orange'
set arrow from 1607.994876,0 to 1607.994876,0.000720562 nohead lc rgb '#666666' lw 1
set arrow from 1587.515193,0 to 1587.515193,0.000720562 nohead lc rgb '#666666' lw 1
set arrow from 1575.651741,0 to 1575.651741,0.000720562 nohead lc rgb '#666666' lw 1
set arrow from 1609.318074,0 to 1609.318074,0.000720562 nohead lc rgb '#666666' lw 1
set arrow from 1575.244074,0 to 1575.244074,0.000720562 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XL/graph500-27/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XL/graph500-27/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
