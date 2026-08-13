set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/XL/graph500-26/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / graph500-26 (XL) — max_mem_24G workers_4\nmedian=449.237s  mean=455.802s  std=10.823s  min=448.362s  max=473.384s  p90=467.724s  p95=470.554s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0561732]
set xrange [424.415107:497.330757]
set grid y
set key top right
set arrow from 449.236513,0 to 449.236513,0.0561732 nohead lc rgb 'red' lw 2
set label 'median 449.237s' at 449.236513,0.0561732 offset char 0,1 tc rgb 'red'
set arrow from 467.724166,0 to 467.724166,0.0561732 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 467.724s' at 467.724166,0.0561732 offset char 0,1 tc rgb 'orange'
set arrow from 470.554161,0 to 470.554161,0.0561732 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 470.554s' at 470.554161,0.0561732 offset char 0,1 tc rgb 'orange'
set arrow from 448.792347,0 to 448.792347,0.00172841 nohead lc rgb '#666666' lw 1
set arrow from 473.384155,0 to 473.384155,0.00172841 nohead lc rgb '#666666' lw 1
set arrow from 449.236513,0 to 449.236513,0.00172841 nohead lc rgb '#666666' lw 1
set arrow from 459.234182,0 to 459.234182,0.00172841 nohead lc rgb '#666666' lw 1
set arrow from 448.361709,0 to 448.361709,0.00172841 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
