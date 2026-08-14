set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/mis/XS/cit-Patents/max_mem_24G_workers_4/wall_time.png'
set title "mis / cit-Patents (XS) — max_mem_24G workers_4\nmedian=133.187s  mean=133.311s  std=0.999s  min=131.862s  max=134.618s  p90=134.251s  p95=134.434s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.796556]
set xrange [130.681901:135.798082]
set grid y
set key top right
set arrow from 133.186620,0 to 133.186620,0.796556 nohead lc rgb 'red' lw 2
set label 'median 133.187s' at 133.186620,0.796556 offset char 0,1 tc rgb 'red'
set arrow from 134.250949,0 to 134.250949,0.796556 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 134.251s' at 134.250949,0.796556 offset char 0,1 tc rgb 'orange'
set arrow from 134.434392,0 to 134.434392,0.796556 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 134.434s' at 134.434392,0.796556 offset char 0,1 tc rgb 'orange'
set arrow from 133.186620,0 to 133.186620,0.0245094 nohead lc rgb '#666666' lw 1
set arrow from 133.185976,0 to 133.185976,0.0245094 nohead lc rgb '#666666' lw 1
set arrow from 131.862147,0 to 131.862147,0.0245094 nohead lc rgb '#666666' lw 1
set arrow from 133.700619,0 to 133.700619,0.0245094 nohead lc rgb '#666666' lw 1
set arrow from 134.617836,0 to 134.617836,0.0245094 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/mis/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/mis/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
