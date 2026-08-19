set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.png'
set title "wcc / datagen-sf3k-fb (XL) — max_mem_24G workers_4\nmedian=4272.040s  mean=4269.700s  std=47.832s  min=4199.817s  max=4331.329s  p90=4313.839s  p95=4322.584s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0143429]
set xrange [4131.260977:4399.885140]
set grid y
set key top right
set arrow from 4272.039787,0 to 4272.039787,0.0143429 nohead lc rgb 'red' lw 2
set label 'median 4272.040s' at 4272.039787,0.0143429 offset char 0,1 tc rgb 'red'
set arrow from 4313.839307,0 to 4313.839307,0.0143429 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 4313.839s' at 4313.839307,0.0143429 offset char 0,1 tc rgb 'orange'
set arrow from 4322.584185,0 to 4322.584185,0.0143429 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 4322.584s' at 4322.584185,0.0143429 offset char 0,1 tc rgb 'orange'
set arrow from 4272.039787,0 to 4272.039787,0.000441319 nohead lc rgb '#666666' lw 1
set arrow from 4199.817054,0 to 4199.817054,0.000441319 nohead lc rgb '#666666' lw 1
set arrow from 4257.711026,0 to 4257.711026,0.000441319 nohead lc rgb '#666666' lw 1
set arrow from 4287.604673,0 to 4287.604673,0.000441319 nohead lc rgb '#666666' lw 1
set arrow from 4331.329063,0 to 4331.329063,0.000441319 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
