set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/mis/XL/graph500-26/max_mem_30G_workers_16/wall_time.png'
set title "mis / graph500-26 (XL) — max_mem_30G workers_16\nmedian=506.157s  mean=508.322s  std=14.752s  min=493.062s  max=529.585s  p90=524.038s  p95=526.811s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0291876]
set xrange [450.343378:572.303106]
set grid y
set key top right
set arrow from 506.156923,0 to 506.156923,0.0291876 nohead lc rgb 'red' lw 2
set label 'median 506.157s' at 506.156923,0.0291876 offset char 0,1 tc rgb 'red'
set arrow from 524.037817,0 to 524.037817,0.0291876 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 524.038s' at 524.037817,0.0291876 offset char 0,1 tc rgb 'orange'
set arrow from 526.811221,0 to 526.811221,0.0291876 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 526.811s' at 526.811221,0.0291876 offset char 0,1 tc rgb 'orange'
set arrow from 493.061859,0 to 493.061859,0.000898081 nohead lc rgb '#666666' lw 1
set arrow from 506.156923,0 to 506.156923,0.000898081 nohead lc rgb '#666666' lw 1
set arrow from 529.584625,0 to 529.584625,0.000898081 nohead lc rgb '#666666' lw 1
set arrow from 497.090355,0 to 497.090355,0.000898081 nohead lc rgb '#666666' lw 1
set arrow from 515.717605,0 to 515.717605,0.000898081 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/mis/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/mis/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
