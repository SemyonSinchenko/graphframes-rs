set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XS/cit-Patents/max_mem_24G_workers_4/wall_time.png'
set title "wcc / cit-Patents (XS) — max_mem_24G workers_4\nmedian=16.571s  mean=16.582s  std=0.118s  min=16.428s  max=16.752s  p90=16.699s  p95=16.726s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:5.31295]
set xrange [16.242817:16.937575]
set grid y
set key top right
set arrow from 16.571362,0 to 16.571362,5.31295 nohead lc rgb 'red' lw 2
set label 'median 16.571s' at 16.571362,5.31295 offset char 0,1 tc rgb 'red'
set arrow from 16.698748,0 to 16.698748,5.31295 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 16.699s' at 16.698748,5.31295 offset char 0,1 tc rgb 'orange'
set arrow from 16.725504,0 to 16.725504,5.31295 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 16.726s' at 16.725504,5.31295 offset char 0,1 tc rgb 'orange'
set arrow from 16.571362,0 to 16.571362,0.163475 nohead lc rgb '#666666' lw 1
set arrow from 16.618480,0 to 16.618480,0.163475 nohead lc rgb '#666666' lw 1
set arrow from 16.752260,0 to 16.752260,0.163475 nohead lc rgb '#666666' lw 1
set arrow from 16.537674,0 to 16.537674,0.163475 nohead lc rgb '#666666' lw 1
set arrow from 16.428132,0 to 16.428132,0.163475 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
