set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/mis/XL/graph500-27/max_mem_30G_workers_16/wall_time.png'
set title "mis / graph500-27 (XL) — max_mem_30G workers_16\nmedian=2790.287s  mean=2849.134s  std=219.750s  min=2593.031s  max=3195.215s  p90=3068.112s  p95=3131.663s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0046759]
set xrange [2391.729990:3396.515863]
set grid y
set key top right
set arrow from 2790.287110,0 to 2790.287110,0.0046759 nohead lc rgb 'red' lw 2
set label 'median 2790.287s' at 2790.287110,0.0046759 offset char 0,1 tc rgb 'red'
set arrow from 3068.111665,0 to 3068.111665,0.0046759 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 3068.112s' at 3068.111665,0.0046759 offset char 0,1 tc rgb 'orange'
set arrow from 3131.663360,0 to 3131.663360,0.0046759 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 3131.663s' at 3131.663360,0.0046759 offset char 0,1 tc rgb 'orange'
set arrow from 3195.215055,0 to 3195.215055,0.000143874 nohead lc rgb '#666666' lw 1
set arrow from 2877.456581,0 to 2877.456581,0.000143874 nohead lc rgb '#666666' lw 1
set arrow from 2790.287110,0 to 2790.287110,0.000143874 nohead lc rgb '#666666' lw 1
set arrow from 2593.030798,0 to 2593.030798,0.000143874 nohead lc rgb '#666666' lw 1
set arrow from 2789.680040,0 to 2789.680040,0.000143874 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/mis/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/mis/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
