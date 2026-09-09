set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/2XL/graph500-29/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / graph500-29 (2XL) — max_mem_24G workers_4\nmedian=4821.330s  mean=4852.910s  std=105.899s  min=4749.378s  max=5026.924s  p90=4962.512s  p95=4994.718s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.00702923]
set xrange [4600.604392:5175.698018]
set grid y
set key top right
set arrow from 4821.330156,0 to 4821.330156,0.00702923 nohead lc rgb 'red' lw 2
set label 'median 4821.330s' at 4821.330156,0.00702923 offset char 0,1 tc rgb 'red'
set arrow from 4962.512265,0 to 4962.512265,0.00702923 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 4962.512s' at 4962.512265,0.00702923 offset char 0,1 tc rgb 'orange'
set arrow from 4994.718211,0 to 4994.718211,0.00702923 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 4994.718s' at 4994.718211,0.00702923 offset char 0,1 tc rgb 'orange'
set arrow from 4801.022083,0 to 4801.022083,0.000216284 nohead lc rgb '#666666' lw 1
set arrow from 4865.894425,0 to 4865.894425,0.000216284 nohead lc rgb '#666666' lw 1
set arrow from 4749.378253,0 to 4749.378253,0.000216284 nohead lc rgb '#666666' lw 1
set arrow from 4821.330156,0 to 4821.330156,0.000216284 nohead lc rgb '#666666' lw 1
set arrow from 5026.924158,0 to 5026.924158,0.000216284 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/2XL/graph500-29/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/2XL/graph500-29/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
