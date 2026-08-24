set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/graph500-26/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-26 (XL) — max_mem_24G workers_4\nmedian=776.355s  mean=780.043s  std=15.387s  min=768.123s  max=806.612s  p90=795.213s  p95=800.913s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.065279]
set xrange [751.824938:822.910164]
set grid y
set key top right
set arrow from 776.355293,0 to 776.355293,0.065279 nohead lc rgb 'red' lw 2
set label 'median 776.355s' at 776.355293,0.065279 offset char 0,1 tc rgb 'red'
set arrow from 795.213222,0 to 795.213222,0.065279 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 795.213s' at 795.213222,0.065279 offset char 0,1 tc rgb 'orange'
set arrow from 800.912836,0 to 800.912836,0.065279 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 800.913s' at 800.912836,0.065279 offset char 0,1 tc rgb 'orange'
set arrow from 778.114382,0 to 778.114382,0.00200858 nohead lc rgb '#666666' lw 1
set arrow from 771.007818,0 to 771.007818,0.00200858 nohead lc rgb '#666666' lw 1
set arrow from 776.355293,0 to 776.355293,0.00200858 nohead lc rgb '#666666' lw 1
set arrow from 768.122654,0 to 768.122654,0.00200858 nohead lc rgb '#666666' lw 1
set arrow from 806.612449,0 to 806.612449,0.00200858 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
