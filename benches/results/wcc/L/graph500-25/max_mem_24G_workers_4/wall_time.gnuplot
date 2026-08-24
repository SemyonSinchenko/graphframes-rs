set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/L/graph500-25/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-25 (L) — max_mem_24G workers_4\nmedian=345.971s  mean=347.068s  std=3.380s  min=344.547s  max=353.015s  p90=350.238s  p95=351.626s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:1.25008]
set xrange [343.768259:353.793924]
set grid y
set key top right
set arrow from 345.970883,0 to 345.970883,1.25008 nohead lc rgb 'red' lw 2
set label 'median 345.971s' at 345.970883,1.25008 offset char 0,1 tc rgb 'red'
set arrow from 350.238056,0 to 350.238056,1.25008 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 350.238s' at 350.238056,1.25008 offset char 0,1 tc rgb 'orange'
set arrow from 351.626466,0 to 351.626466,1.25008 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 351.626s' at 351.626466,1.25008 offset char 0,1 tc rgb 'orange'
set arrow from 353.014877,0 to 353.014877,0.038464 nohead lc rgb '#666666' lw 1
set arrow from 346.072824,0 to 346.072824,0.038464 nohead lc rgb '#666666' lw 1
set arrow from 345.733123,0 to 345.733123,0.038464 nohead lc rgb '#666666' lw 1
set arrow from 344.547306,0 to 344.547306,0.038464 nohead lc rgb '#666666' lw 1
set arrow from 345.970883,0 to 345.970883,0.038464 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/L/graph500-25/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/L/graph500-25/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
