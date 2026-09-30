set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/mis/L/graph500-25/max_mem_30G_workers_16/wall_time.png'
set title "mis / graph500-25 (L) — max_mem_30G workers_16\nmedian=242.205s  mean=240.209s  std=4.537s  min=233.968s  max=244.654s  p90=244.090s  p95=244.372s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0933154]
set xrange [220.023898:258.597375]
set grid y
set key top right
set arrow from 242.205417,0 to 242.205417,0.0933154 nohead lc rgb 'red' lw 2
set label 'median 242.205s' at 242.205417,0.0933154 offset char 0,1 tc rgb 'red'
set arrow from 244.089629,0 to 244.089629,0.0933154 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 244.090s' at 244.089629,0.0933154 offset char 0,1 tc rgb 'orange'
set arrow from 244.371660,0 to 244.371660,0.0933154 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 244.372s' at 244.371660,0.0933154 offset char 0,1 tc rgb 'orange'
set arrow from 242.205417,0 to 242.205417,0.00287124 nohead lc rgb '#666666' lw 1
set arrow from 236.976852,0 to 236.976852,0.00287124 nohead lc rgb '#666666' lw 1
set arrow from 244.653691,0 to 244.653691,0.00287124 nohead lc rgb '#666666' lw 1
set arrow from 243.243537,0 to 243.243537,0.00287124 nohead lc rgb '#666666' lw 1
set arrow from 233.967581,0 to 233.967581,0.00287124 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/mis/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/mis/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
