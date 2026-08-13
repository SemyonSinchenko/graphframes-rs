set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/sp/XS/cit-Patents/max_mem_24G_workers_4/wall_time.png'
set title "sp / cit-Patents (XS) — max_mem_24G workers_4\nmedian=2.470s  mean=2.497s  std=0.068s  min=2.451s  max=2.618s  p90=2.563s  p95=2.590s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:27.7667]
set xrange [2.413983:2.654911]
set grid y
set key top right
set arrow from 2.470235,0 to 2.470235,27.7667 nohead lc rgb 'red' lw 2
set label 'median 2.470s' at 2.470235,27.7667 offset char 0,1 tc rgb 'red'
set arrow from 2.562892,0 to 2.562892,27.7667 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 2.563s' at 2.562892,27.7667 offset char 0,1 tc rgb 'orange'
set arrow from 2.590241,0 to 2.590241,27.7667 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 2.590s' at 2.590241,27.7667 offset char 0,1 tc rgb 'orange'
set arrow from 2.470235,0 to 2.470235,0.85436 nohead lc rgb '#666666' lw 1
set arrow from 2.617590,0 to 2.617590,0.85436 nohead lc rgb '#666666' lw 1
set arrow from 2.464572,0 to 2.464572,0.85436 nohead lc rgb '#666666' lw 1
set arrow from 2.451304,0 to 2.451304,0.85436 nohead lc rgb '#666666' lw 1
set arrow from 2.480845,0 to 2.480845,0.85436 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/sp/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/sp/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
