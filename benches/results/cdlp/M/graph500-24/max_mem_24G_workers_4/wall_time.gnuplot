set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/cdlp/M/graph500-24/max_mem_24G_workers_4/wall_time.png'
set title "cdlp / graph500-24 (M) — max_mem_24G workers_4\nmedian=1015.166s  mean=1032.904s  std=45.576s  min=992.599s  max=1110.501s  p90=1079.097s  p95=1094.799s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0239121]
set xrange [951.949267:1151.150195]
set grid y
set key top right
set arrow from 1015.165781,0 to 1015.165781,0.0239121 nohead lc rgb 'red' lw 2
set label 'median 1015.166s' at 1015.165781,0.0239121 offset char 0,1 tc rgb 'red'
set arrow from 1079.096847,0 to 1079.096847,0.0239121 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1079.097s' at 1079.096847,0.0239121 offset char 0,1 tc rgb 'orange'
set arrow from 1094.798896,0 to 1094.798896,0.0239121 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1094.799s' at 1094.798896,0.0239121 offset char 0,1 tc rgb 'orange'
set arrow from 1031.990702,0 to 1031.990702,0.000735757 nohead lc rgb '#666666' lw 1
set arrow from 992.598516,0 to 992.598516,0.000735757 nohead lc rgb '#666666' lw 1
set arrow from 1015.165781,0 to 1015.165781,0.000735757 nohead lc rgb '#666666' lw 1
set arrow from 1014.265733,0 to 1014.265733,0.000735757 nohead lc rgb '#666666' lw 1
set arrow from 1110.500945,0 to 1110.500945,0.000735757 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/cdlp/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/cdlp/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
