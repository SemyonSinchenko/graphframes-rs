set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pic/L/graph500-25/max_mem_30G_workers_16/wall_time.png'
set title "pic / graph500-25 (L) — max_mem_30G workers_16\nmedian=404.838s  mean=405.612s  std=2.199s  min=403.498s  max=408.704s  p90=408.039s  p95=408.372s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.192754]
set xrange [396.739672:415.462190]
set grid y
set key top right
set arrow from 404.837772,0 to 404.837772,0.192754 nohead lc rgb 'red' lw 2
set label 'median 404.838s' at 404.837772,0.192754 offset char 0,1 tc rgb 'red'
set arrow from 408.039187,0 to 408.039187,0.192754 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 408.039s' at 408.039187,0.192754 offset char 0,1 tc rgb 'orange'
set arrow from 408.371523,0 to 408.371523,0.192754 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 408.372s' at 408.371523,0.192754 offset char 0,1 tc rgb 'orange'
set arrow from 403.498003,0 to 403.498003,0.0059309 nohead lc rgb '#666666' lw 1
set arrow from 403.977142,0 to 403.977142,0.0059309 nohead lc rgb '#666666' lw 1
set arrow from 408.703859,0 to 408.703859,0.0059309 nohead lc rgb '#666666' lw 1
set arrow from 407.042179,0 to 407.042179,0.0059309 nohead lc rgb '#666666' lw 1
set arrow from 404.837772,0 to 404.837772,0.0059309 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pic/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pic/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
