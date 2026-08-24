set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/XL/datagen-9_3-zf/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / datagen-9_3-zf (XL) — max_mem_24G workers_4\nmedian=3331.165s  mean=3343.601s  std=54.187s  min=3286.714s  max=3432.169s  p90=3397.989s  p95=3415.079s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0168348]
set xrange [3228.280900:3490.602651]
set grid y
set key top right
set arrow from 3331.164560,0 to 3331.164560,0.0168348 nohead lc rgb 'red' lw 2
set label 'median 3331.165s' at 3331.164560,0.0168348 offset char 0,1 tc rgb 'red'
set arrow from 3397.989203,0 to 3397.989203,0.0168348 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 3397.989s' at 3397.989203,0.0168348 offset char 0,1 tc rgb 'orange'
set arrow from 3415.079289,0 to 3415.079289,0.0168348 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 3415.079s' at 3415.079289,0.0168348 offset char 0,1 tc rgb 'orange'
set arrow from 3432.169375,0 to 3432.169375,0.000517992 nohead lc rgb '#666666' lw 1
set arrow from 3286.714176,0 to 3286.714176,0.000517992 nohead lc rgb '#666666' lw 1
set arrow from 3321.239311,0 to 3321.239311,0.000517992 nohead lc rgb '#666666' lw 1
set arrow from 3331.164560,0 to 3331.164560,0.000517992 nohead lc rgb '#666666' lw 1
set arrow from 3346.718945,0 to 3346.718945,0.000517992 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/XL/datagen-9_3-zf/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/XL/datagen-9_3-zf/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
