set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/M/graph500-24/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / graph500-24 (M) — max_mem_24G workers_4\nmedian=92.753s  mean=94.944s  std=3.764s  min=92.023s  max=100.957s  p90=99.108s  p95=100.032s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.156221]
set xrange [83.580796:109.399092]
set grid y
set key top right
set arrow from 92.753087,0 to 92.753087,0.156221 nohead lc rgb 'red' lw 2
set label 'median 92.753s' at 92.753087,0.156221 offset char 0,1 tc rgb 'red'
set arrow from 99.107543,0 to 99.107543,0.156221 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 99.108s' at 99.107543,0.156221 offset char 0,1 tc rgb 'orange'
set arrow from 100.032253,0 to 100.032253,0.156221 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 100.032s' at 100.032253,0.156221 offset char 0,1 tc rgb 'orange'
set arrow from 92.022925,0 to 92.022925,0.00480679 nohead lc rgb '#666666' lw 1
set arrow from 100.956963,0 to 100.956963,0.00480679 nohead lc rgb '#666666' lw 1
set arrow from 96.333414,0 to 96.333414,0.00480679 nohead lc rgb '#666666' lw 1
set arrow from 92.652252,0 to 92.652252,0.00480679 nohead lc rgb '#666666' lw 1
set arrow from 92.753087,0 to 92.753087,0.00480679 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
