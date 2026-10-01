set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/sp/XL/graph500-27/max_mem_30G_workers_16/wall_time.png'
set title "sp / graph500-27 (XL) — max_mem_30G workers_16\nmedian=182.616s  mean=181.996s  std=1.339s  min=180.131s  max=183.398s  p90=183.133s  p95=183.266s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.33329]
set xrange [176.383608:187.145253]
set grid y
set key top right
set arrow from 182.615604,0 to 182.615604,0.33329 nohead lc rgb 'red' lw 2
set label 'median 182.616s' at 182.615604,0.33329 offset char 0,1 tc rgb 'red'
set arrow from 183.133137,0 to 183.133137,0.33329 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 183.133s' at 183.133137,0.33329 offset char 0,1 tc rgb 'orange'
set arrow from 183.265505,0 to 183.265505,0.33329 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 183.266s' at 183.265505,0.33329 offset char 0,1 tc rgb 'orange'
set arrow from 182.736033,0 to 182.736033,0.0102551 nohead lc rgb '#666666' lw 1
set arrow from 182.615604,0 to 182.615604,0.0102551 nohead lc rgb '#666666' lw 1
set arrow from 180.130989,0 to 180.130989,0.0102551 nohead lc rgb '#666666' lw 1
set arrow from 183.397873,0 to 183.397873,0.0102551 nohead lc rgb '#666666' lw 1
set arrow from 181.102000,0 to 181.102000,0.0102551 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/sp/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/sp/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
