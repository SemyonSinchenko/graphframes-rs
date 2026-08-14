set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / datagen-sf3k-fb (XL) — max_mem_24G workers_4\nmedian=2203.234s  mean=2212.150s  std=51.097s  min=2158.366s  max=2292.662s  p90=2265.046s  p95=2278.854s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0114475]
set xrange [2064.908408:2386.120014]
set grid y
set key top right
set arrow from 2203.233830,0 to 2203.233830,0.0114475 nohead lc rgb 'red' lw 2
set label 'median 2203.234s' at 2203.233830,0.0114475 offset char 0,1 tc rgb 'red'
set arrow from 2265.045548,0 to 2265.045548,0.0114475 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 2265.046s' at 2265.045548,0.0114475 offset char 0,1 tc rgb 'orange'
set arrow from 2278.853948,0 to 2278.853948,0.0114475 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 2278.854s' at 2278.853948,0.0114475 offset char 0,1 tc rgb 'orange'
set arrow from 2158.366074,0 to 2158.366074,0.00035223 nohead lc rgb '#666666' lw 1
set arrow from 2223.620348,0 to 2223.620348,0.00035223 nohead lc rgb '#666666' lw 1
set arrow from 2182.868447,0 to 2182.868447,0.00035223 nohead lc rgb '#666666' lw 1
set arrow from 2203.233830,0 to 2203.233830,0.00035223 nohead lc rgb '#666666' lw 1
set arrow from 2292.662349,0 to 2292.662349,0.00035223 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
