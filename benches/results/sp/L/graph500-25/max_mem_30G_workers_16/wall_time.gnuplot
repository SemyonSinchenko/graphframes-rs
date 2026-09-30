set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/sp/L/graph500-25/max_mem_30G_workers_16/wall_time.png'
set title "sp / graph500-25 (L) — max_mem_30G workers_16\nmedian=26.436s  mean=26.147s  std=0.948s  min=24.652s  max=27.126s  p90=26.932s  p95=27.029s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.633946]
set xrange [22.901907:28.875897]
set grid y
set key top right
set arrow from 26.435934,0 to 26.435934,0.633946 nohead lc rgb 'red' lw 2
set label 'median 26.436s' at 26.435934,0.633946 offset char 0,1 tc rgb 'red'
set arrow from 26.932154,0 to 26.932154,0.633946 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 26.932s' at 26.932154,0.633946 offset char 0,1 tc rgb 'orange'
set arrow from 27.028966,0 to 27.028966,0.633946 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 27.029s' at 27.028966,0.633946 offset char 0,1 tc rgb 'orange'
set arrow from 25.878585,0 to 25.878585,0.019506 nohead lc rgb '#666666' lw 1
set arrow from 27.125777,0 to 27.125777,0.019506 nohead lc rgb '#666666' lw 1
set arrow from 24.652027,0 to 24.652027,0.019506 nohead lc rgb '#666666' lw 1
set arrow from 26.641719,0 to 26.641719,0.019506 nohead lc rgb '#666666' lw 1
set arrow from 26.435934,0 to 26.435934,0.019506 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/sp/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/sp/L/graph500-25/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
