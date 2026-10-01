set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pagerank/XL/graph500-26/max_mem_30G_workers_16/wall_time.png'
set title "pagerank / graph500-26 (XL) — max_mem_30G workers_16\nmedian=148.680s  mean=148.345s  std=2.672s  min=145.800s  max=152.439s  p90=150.949s  p95=151.694s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.181826]
set xrange [139.788805:158.450550]
set grid y
set key top right
set arrow from 148.679653,0 to 148.679653,0.181826 nohead lc rgb 'red' lw 2
set label 'median 148.680s' at 148.679653,0.181826 offset char 0,1 tc rgb 'red'
set arrow from 150.948996,0 to 150.948996,0.181826 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 150.949s' at 150.948996,0.181826 offset char 0,1 tc rgb 'orange'
set arrow from 151.693946,0 to 151.693946,0.181826 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 151.694s' at 151.693946,0.181826 offset char 0,1 tc rgb 'orange'
set arrow from 145.800459,0 to 145.800459,0.00559464 nohead lc rgb '#666666' lw 1
set arrow from 146.092784,0 to 146.092784,0.00559464 nohead lc rgb '#666666' lw 1
set arrow from 148.679653,0 to 148.679653,0.00559464 nohead lc rgb '#666666' lw 1
set arrow from 152.438896,0 to 152.438896,0.00559464 nohead lc rgb '#666666' lw 1
set arrow from 148.714146,0 to 148.714146,0.00559464 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pagerank/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pagerank/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
