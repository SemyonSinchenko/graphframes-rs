set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/wcc/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "wcc / graph500-24 (M) — max_mem_30G workers_16\nmedian=33.275s  mean=33.270s  std=0.083s  min=33.163s  max=33.359s  p90=33.351s  p95=33.355s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:4.69991]
set xrange [32.907116:33.614766]
set grid y
set key top right
set arrow from 33.275283,0 to 33.275283,4.69991 nohead lc rgb 'red' lw 2
set label 'median 33.275s' at 33.275283,4.69991 offset char 0,1 tc rgb 'red'
set arrow from 33.351248,0 to 33.351248,4.69991 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 33.351s' at 33.351248,4.69991 offset char 0,1 tc rgb 'orange'
set arrow from 33.355249,0 to 33.355249,4.69991 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 33.355s' at 33.355249,4.69991 offset char 0,1 tc rgb 'orange'
set arrow from 33.339244,0 to 33.339244,0.144613 nohead lc rgb '#666666' lw 1
set arrow from 33.162631,0 to 33.162631,0.144613 nohead lc rgb '#666666' lw 1
set arrow from 33.359251,0 to 33.359251,0.144613 nohead lc rgb '#666666' lw 1
set arrow from 33.275283,0 to 33.275283,0.144613 nohead lc rgb '#666666' lw 1
set arrow from 33.212228,0 to 33.212228,0.144613 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/wcc/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/wcc/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
