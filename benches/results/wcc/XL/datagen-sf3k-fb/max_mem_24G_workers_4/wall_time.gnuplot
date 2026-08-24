set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.png'
set title "wcc / datagen-sf3k-fb (XL) — max_mem_24G workers_4\nmedian=2241.037s  mean=2236.641s  std=20.433s  min=2216.590s  max=2266.023s  p90=2256.352s  p95=2261.187s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0206712]
set xrange [2161.242186:2321.370442]
set grid y
set key top right
set arrow from 2241.037056,0 to 2241.037056,0.0206712 nohead lc rgb 'red' lw 2
set label 'median 2241.037s' at 2241.037056,0.0206712 offset char 0,1 tc rgb 'red'
set arrow from 2256.352041,0 to 2256.352041,0.0206712 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 2256.352s' at 2256.352041,0.0206712 offset char 0,1 tc rgb 'orange'
set arrow from 2261.187390,0 to 2261.187390,0.0206712 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 2261.187s' at 2261.187390,0.0206712 offset char 0,1 tc rgb 'orange'
set arrow from 2241.037056,0 to 2241.037056,0.000636038 nohead lc rgb '#666666' lw 1
set arrow from 2216.589890,0 to 2216.589890,0.000636038 nohead lc rgb '#666666' lw 1
set arrow from 2241.845996,0 to 2241.845996,0.000636038 nohead lc rgb '#666666' lw 1
set arrow from 2217.711816,0 to 2217.711816,0.000636038 nohead lc rgb '#666666' lw 1
set arrow from 2266.022738,0 to 2266.022738,0.000636038 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
