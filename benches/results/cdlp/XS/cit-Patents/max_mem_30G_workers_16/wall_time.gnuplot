set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/cdlp/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "cdlp / cit-Patents (XS) — max_mem_30G workers_16\nmedian=12.101s  mean=12.094s  std=0.052s  min=12.022s  max=12.164s  p90=12.141s  p95=12.152s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:13.1776]
set xrange [11.948031:12.237259]
set grid y
set key top right
set arrow from 12.100922,0 to 12.100922,13.1776 nohead lc rgb 'red' lw 2
set label 'median 12.101s' at 12.100922,13.1776 offset char 0,1 tc rgb 'red'
set arrow from 12.141321,0 to 12.141321,13.1776 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 12.141s' at 12.141321,13.1776 offset char 0,1 tc rgb 'orange'
set arrow from 12.152486,0 to 12.152486,13.1776 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 12.152s' at 12.152486,13.1776 offset char 0,1 tc rgb 'orange'
set arrow from 12.100922,0 to 12.100922,0.405466 nohead lc rgb '#666666' lw 1
set arrow from 12.021639,0 to 12.021639,0.405466 nohead lc rgb '#666666' lw 1
set arrow from 12.075730,0 to 12.075730,0.405466 nohead lc rgb '#666666' lw 1
set arrow from 12.107827,0 to 12.107827,0.405466 nohead lc rgb '#666666' lw 1
set arrow from 12.163651,0 to 12.163651,0.405466 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/cdlp/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/cdlp/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
