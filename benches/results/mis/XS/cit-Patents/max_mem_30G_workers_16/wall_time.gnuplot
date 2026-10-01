set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/mis/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "mis / cit-Patents (XS) — max_mem_30G workers_16\nmedian=9.686s  mean=9.698s  std=0.398s  min=9.147s  max=10.271s  p90=10.046s  p95=10.158s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:16.0172]
set xrange [9.087033:10.331834]
set grid y
set key top right
set arrow from 9.685915,0 to 9.685915,16.0172 nohead lc rgb 'red' lw 2
set label 'median 9.686s' at 9.685915,16.0172 offset char 0,1 tc rgb 'red'
set arrow from 10.045577,0 to 10.045577,16.0172 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 10.046s' at 10.045577,16.0172 offset char 0,1 tc rgb 'orange'
set arrow from 10.158488,0 to 10.158488,16.0172 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 10.158s' at 10.158488,16.0172 offset char 0,1 tc rgb 'orange'
set arrow from 9.685915,0 to 9.685915,0.492836 nohead lc rgb '#666666' lw 1
set arrow from 10.271400,0 to 10.271400,0.492836 nohead lc rgb '#666666' lw 1
set arrow from 9.147468,0 to 9.147468,0.492836 nohead lc rgb '#666666' lw 1
set arrow from 9.680491,0 to 9.680491,0.492836 nohead lc rgb '#666666' lw 1
set arrow from 9.706843,0 to 9.706843,0.492836 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/mis/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/mis/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
