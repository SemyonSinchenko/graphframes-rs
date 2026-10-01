set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pagerank/L/graph500-25/max_mem_30G_workers_16/wall_time.png'
set title "pagerank / graph500-25 (L) — max_mem_30G workers_16\nmedian=62.320s  mean=62.062s  std=0.727s  min=61.203s  max=62.994s  p90=62.731s  p95=62.862s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.561812]
set xrange [59.194206:65.002740]
set grid y
set key top right
set arrow from 62.319569,0 to 62.319569,0.561812 nohead lc rgb 'red' lw 2
set label 'median 62.320s' at 62.319569,0.561812 offset char 0,1 tc rgb 'red'
set arrow from 62.730597,0 to 62.730597,0.561812 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 62.731s' at 62.730597,0.561812 offset char 0,1 tc rgb 'orange'
set arrow from 62.862460,0 to 62.862460,0.561812 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 62.862s' at 62.862460,0.561812 offset char 0,1 tc rgb 'orange'
set arrow from 62.994323,0 to 62.994323,0.0172865 nohead lc rgb '#666666' lw 1
set arrow from 62.319569,0 to 62.319569,0.0172865 nohead lc rgb '#666666' lw 1
set arrow from 62.335009,0 to 62.335009,0.0172865 nohead lc rgb '#666666' lw 1
set arrow from 61.459246,0 to 61.459246,0.0172865 nohead lc rgb '#666666' lw 1
set arrow from 61.202623,0 to 61.202623,0.0172865 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pagerank/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pagerank/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
