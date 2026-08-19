set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/sp/M/graph500-24/max_mem_24G_workers_4/wall_time.png'
set title "sp / graph500-24 (M) — max_mem_24G workers_4\nmedian=98.477s  mean=98.973s  std=1.923s  min=96.853s  max=100.985s  p90=100.983s  p95=100.984s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.204178]
set xrange [90.943962:106.893644]
set grid y
set key top right
set arrow from 98.477010,0 to 98.477010,0.204178 nohead lc rgb 'red' lw 2
set label 'median 98.477s' at 98.477010,0.204178 offset char 0,1 tc rgb 'red'
set arrow from 100.982718,0 to 100.982718,0.204178 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 100.983s' at 100.982718,0.204178 offset char 0,1 tc rgb 'orange'
set arrow from 100.983841,0 to 100.983841,0.204178 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 100.984s' at 100.983841,0.204178 offset char 0,1 tc rgb 'orange'
set arrow from 100.984965,0 to 100.984965,0.0062824 nohead lc rgb '#666666' lw 1
set arrow from 100.979348,0 to 100.979348,0.0062824 nohead lc rgb '#666666' lw 1
set arrow from 96.852641,0 to 96.852641,0.0062824 nohead lc rgb '#666666' lw 1
set arrow from 97.568643,0 to 97.568643,0.0062824 nohead lc rgb '#666666' lw 1
set arrow from 98.477010,0 to 98.477010,0.0062824 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/sp/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/sp/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
