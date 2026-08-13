set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/cdlp/XL/graph500-26/max_mem_24G_workers_4/wall_time.png'
set title "cdlp / graph500-26 (XL) — max_mem_24G workers_4\nmedian=3555.284s  mean=3552.479s  std=135.976s  min=3430.059s  max=3771.965s  p90=3685.741s  p95=3728.853s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.00430637]
set xrange [3183.016122:4019.007811]
set grid y
set key top right
set arrow from 3555.283531,0 to 3555.283531,0.00430637 nohead lc rgb 'red' lw 2
set label 'median 3555.284s' at 3555.283531,0.00430637 offset char 0,1 tc rgb 'red'
set arrow from 3685.741106,0 to 3685.741106,0.00430637 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 3685.741s' at 3685.741106,0.00430637 offset char 0,1 tc rgb 'orange'
set arrow from 3728.852956,0 to 3728.852956,0.00430637 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 3728.853s' at 3728.852956,0.00430637 offset char 0,1 tc rgb 'orange'
set arrow from 3556.405556,0 to 3556.405556,0.000132504 nohead lc rgb '#666666' lw 1
set arrow from 3448.683284,0 to 3448.683284,0.000132504 nohead lc rgb '#666666' lw 1
set arrow from 3430.059128,0 to 3430.059128,0.000132504 nohead lc rgb '#666666' lw 1
set arrow from 3555.283531,0 to 3555.283531,0.000132504 nohead lc rgb '#666666' lw 1
set arrow from 3771.964806,0 to 3771.964806,0.000132504 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/cdlp/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/cdlp/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
