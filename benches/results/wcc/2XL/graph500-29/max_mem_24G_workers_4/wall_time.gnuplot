set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/2XL/graph500-29/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-29 (2XL) — max_mem_24G workers_4\nmedian=7128.812s  mean=7135.308s  std=53.225s  min=7063.405s  max=7198.768s  p90=7188.960s  p95=7193.864s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.00774535]
set xrange [6919.061811:7343.111444]
set grid y
set key top right
set arrow from 7128.812114,0 to 7128.812114,0.00774535 nohead lc rgb 'red' lw 2
set label 'median 7128.812s' at 7128.812114,0.00774535 offset char 0,1 tc rgb 'red'
set arrow from 7188.959953,0 to 7188.959953,0.00774535 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 7188.960s' at 7188.959953,0.00774535 offset char 0,1 tc rgb 'orange'
set arrow from 7193.864033,0 to 7193.864033,0.00774535 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 7193.864s' at 7193.864033,0.00774535 offset char 0,1 tc rgb 'orange'
set arrow from 7128.812114,0 to 7128.812114,0.000238319 nohead lc rgb '#666666' lw 1
set arrow from 7063.405142,0 to 7063.405142,0.000238319 nohead lc rgb '#666666' lw 1
set arrow from 7198.768113,0 to 7198.768113,0.000238319 nohead lc rgb '#666666' lw 1
set arrow from 7174.247712,0 to 7174.247712,0.000238319 nohead lc rgb '#666666' lw 1
set arrow from 7111.307288,0 to 7111.307288,0.000238319 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/2XL/graph500-29/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/2XL/graph500-29/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
