set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pic/XL/graph500-27/max_mem_30G_workers_16/wall_time.png'
set title "pic / graph500-27 (XL) — max_mem_30G workers_16\nmedian=1896.833s  mean=1895.249s  std=6.452s  min=1884.474s  max=1901.704s  p90=1900.115s  p95=1900.910s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.191674]
set xrange [1879.356914:1906.821736]
set grid y
set key top right
set arrow from 1896.832775,0 to 1896.832775,0.191674 nohead lc rgb 'red' lw 2
set label 'median 1896.833s' at 1896.832775,0.191674 offset char 0,1 tc rgb 'red'
set arrow from 1900.115060,0 to 1900.115060,0.191674 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1900.115s' at 1900.115060,0.191674 offset char 0,1 tc rgb 'orange'
set arrow from 1900.909632,0 to 1900.909632,0.191674 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1900.910s' at 1900.909632,0.191674 offset char 0,1 tc rgb 'orange'
set arrow from 1901.704204,0 to 1901.704204,0.00589766 nohead lc rgb '#666666' lw 1
set arrow from 1884.474447,0 to 1884.474447,0.00589766 nohead lc rgb '#666666' lw 1
set arrow from 1897.731346,0 to 1897.731346,0.00589766 nohead lc rgb '#666666' lw 1
set arrow from 1895.499862,0 to 1895.499862,0.00589766 nohead lc rgb '#666666' lw 1
set arrow from 1896.832775,0 to 1896.832775,0.00589766 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pic/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pic/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
