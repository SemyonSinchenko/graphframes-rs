set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/wcc/XL/graph500-27/max_mem_30G_workers_16/wall_time.png'
set title "wcc / graph500-27 (XL) — max_mem_30G workers_16\nmedian=480.691s  mean=481.399s  std=1.843s  min=480.034s  max=484.611s  p90=483.243s  p95=483.927s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.672817]
set xrange [478.376775:486.268910]
set grid y
set key top right
set arrow from 480.690524,0 to 480.690524,0.672817 nohead lc rgb 'red' lw 2
set label 'median 480.691s' at 480.690524,0.672817 offset char 0,1 tc rgb 'red'
set arrow from 483.243040,0 to 483.243040,0.672817 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 483.243s' at 483.243040,0.672817 offset char 0,1 tc rgb 'orange'
set arrow from 483.927134,0 to 483.927134,0.672817 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 483.927s' at 483.927134,0.672817 offset char 0,1 tc rgb 'orange'
set arrow from 480.034457,0 to 480.034457,0.020702 nohead lc rgb '#666666' lw 1
set arrow from 480.467931,0 to 480.467931,0.020702 nohead lc rgb '#666666' lw 1
set arrow from 481.190758,0 to 481.190758,0.020702 nohead lc rgb '#666666' lw 1
set arrow from 484.611228,0 to 484.611228,0.020702 nohead lc rgb '#666666' lw 1
set arrow from 480.690524,0 to 480.690524,0.020702 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/wcc/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/wcc/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
