set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/wcc/2XL/graph500-28/max_mem_30G_workers_16/wall_time.png'
set title "wcc / graph500-28 (2XL) — max_mem_30G workers_16\nmedian=1008.907s  mean=1008.888s  std=4.359s  min=1004.443s  max=1015.399s  p90=1013.344s  p95=1014.371s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.105132]
set xrange [993.363001:1026.479047]
set grid y
set key top right
set arrow from 1008.906791,0 to 1008.906791,0.105132 nohead lc rgb 'red' lw 2
set label 'median 1008.907s' at 1008.906791,0.105132 offset char 0,1 tc rgb 'red'
set arrow from 1013.343895,0 to 1013.343895,0.105132 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1013.344s' at 1013.343895,0.105132 offset char 0,1 tc rgb 'orange'
set arrow from 1014.371433,0 to 1014.371433,0.105132 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1014.371s' at 1014.371433,0.105132 offset char 0,1 tc rgb 'orange'
set arrow from 1015.398970,0 to 1015.398970,0.00323482 nohead lc rgb '#666666' lw 1
set arrow from 1005.429852,0 to 1005.429852,0.00323482 nohead lc rgb '#666666' lw 1
set arrow from 1008.906791,0 to 1008.906791,0.00323482 nohead lc rgb '#666666' lw 1
set arrow from 1004.443077,0 to 1004.443077,0.00323482 nohead lc rgb '#666666' lw 1
set arrow from 1010.261283,0 to 1010.261283,0.00323482 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/wcc/2XL/graph500-28/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/wcc/2XL/graph500-28/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
