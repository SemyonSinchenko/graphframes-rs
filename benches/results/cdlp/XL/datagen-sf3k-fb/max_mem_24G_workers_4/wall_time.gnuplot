set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/cdlp/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.png'
set title "cdlp / datagen-sf3k-fb (XL) — max_mem_24G workers_4\nmedian=9343.068s  mean=9329.720s  std=100.829s  min=9192.601s  max=9464.647s  p90=9424.982s  p95=9444.815s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.00521997]
set xrange [9002.969026:9654.278509]
set grid y
set key top right
set arrow from 9343.067800,0 to 9343.067800,0.00521997 nohead lc rgb 'red' lw 2
set label 'median 9343.068s' at 9343.067800,0.00521997 offset char 0,1 tc rgb 'red'
set arrow from 9424.982379,0 to 9424.982379,0.00521997 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 9424.982s' at 9424.982379,0.00521997 offset char 0,1 tc rgb 'orange'
set arrow from 9444.814680,0 to 9444.814680,0.00521997 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 9444.815s' at 9444.814680,0.00521997 offset char 0,1 tc rgb 'orange'
set arrow from 9192.600553,0 to 9192.600553,0.000160614 nohead lc rgb '#666666' lw 1
set arrow from 9282.797284,0 to 9282.797284,0.000160614 nohead lc rgb '#666666' lw 1
set arrow from 9464.646982,0 to 9464.646982,0.000160614 nohead lc rgb '#666666' lw 1
set arrow from 9343.067800,0 to 9343.067800,0.000160614 nohead lc rgb '#666666' lw 1
set arrow from 9365.485475,0 to 9365.485475,0.000160614 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/cdlp/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/cdlp/XL/datagen-sf3k-fb/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
