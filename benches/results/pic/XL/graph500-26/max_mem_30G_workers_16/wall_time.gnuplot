set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pic/XL/graph500-26/max_mem_30G_workers_16/wall_time.png'
set title "pic / graph500-26 (XL) — max_mem_30G workers_16\nmedian=883.478s  mean=883.540s  std=1.999s  min=880.645s  max=885.885s  p90=885.460s  p95=885.673s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.233435]
set xrange [876.160226:890.369910]
set grid y
set key top right
set arrow from 883.478298,0 to 883.478298,0.233435 nohead lc rgb 'red' lw 2
set label 'median 883.478s' at 883.478298,0.233435 offset char 0,1 tc rgb 'red'
set arrow from 885.460262,0 to 885.460262,0.233435 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 885.460s' at 885.460262,0.233435 offset char 0,1 tc rgb 'orange'
set arrow from 885.672513,0 to 885.672513,0.233435 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 885.673s' at 885.672513,0.233435 offset char 0,1 tc rgb 'orange'
set arrow from 885.884763,0 to 885.884763,0.00718261 nohead lc rgb '#666666' lw 1
set arrow from 882.867779,0 to 882.867779,0.00718261 nohead lc rgb '#666666' lw 1
set arrow from 880.645373,0 to 880.645373,0.00718261 nohead lc rgb '#666666' lw 1
set arrow from 883.478298,0 to 883.478298,0.00718261 nohead lc rgb '#666666' lw 1
set arrow from 884.823512,0 to 884.823512,0.00718261 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pic/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pic/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
