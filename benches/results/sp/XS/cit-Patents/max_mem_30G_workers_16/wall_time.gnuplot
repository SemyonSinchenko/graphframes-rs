set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/sp/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "sp / cit-Patents (XS) — max_mem_30G workers_16\nmedian=0.901s  mean=0.897s  std=0.011s  min=0.879s  max=0.906s  p90=0.905s  p95=0.906s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:57.1338]
set xrange [0.856158:0.928356]
set grid y
set key top right
set arrow from 0.900550,0 to 0.900550,57.1338 nohead lc rgb 'red' lw 2
set label 'median 0.901s' at 0.900550,57.1338 offset char 0,1 tc rgb 'red'
set arrow from 0.905293,0 to 0.905293,57.1338 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 0.905s' at 0.905293,57.1338 offset char 0,1 tc rgb 'orange'
set arrow from 0.905533,0 to 0.905533,57.1338 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 0.906s' at 0.905533,57.1338 offset char 0,1 tc rgb 'orange'
set arrow from 0.878740,0 to 0.878740,1.75796 nohead lc rgb '#666666' lw 1
set arrow from 0.905773,0 to 0.905773,1.75796 nohead lc rgb '#666666' lw 1
set arrow from 0.904573,0 to 0.904573,1.75796 nohead lc rgb '#666666' lw 1
set arrow from 0.900550,0 to 0.900550,1.75796 nohead lc rgb '#666666' lw 1
set arrow from 0.894726,0 to 0.894726,1.75796 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/sp/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/sp/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
