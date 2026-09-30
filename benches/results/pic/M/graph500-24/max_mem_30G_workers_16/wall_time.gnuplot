set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pic/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "pic / graph500-24 (M) — max_mem_30G workers_16\nmedian=220.516s  mean=220.365s  std=1.096s  min=218.778s  max=221.770s  p90=221.367s  p95=221.569s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.561974]
set xrange [217.028475:223.519786]
set grid y
set key top right
set arrow from 220.515844,0 to 220.515844,0.561974 nohead lc rgb 'red' lw 2
set label 'median 220.516s' at 220.515844,0.561974 offset char 0,1 tc rgb 'red'
set arrow from 221.367041,0 to 221.367041,0.561974 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 221.367s' at 221.367041,0.561974 offset char 0,1 tc rgb 'orange'
set arrow from 221.568767,0 to 221.568767,0.561974 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 221.569s' at 221.568767,0.561974 offset char 0,1 tc rgb 'orange'
set arrow from 220.515844,0 to 220.515844,0.0172915 nohead lc rgb '#666666' lw 1
set arrow from 220.761860,0 to 220.761860,0.0172915 nohead lc rgb '#666666' lw 1
set arrow from 218.777767,0 to 218.777767,0.0172915 nohead lc rgb '#666666' lw 1
set arrow from 221.770494,0 to 221.770494,0.0172915 nohead lc rgb '#666666' lw 1
set arrow from 219.999087,0 to 219.999087,0.0172915 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pic/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pic/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
