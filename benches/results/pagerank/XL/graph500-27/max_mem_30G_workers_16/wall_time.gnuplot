set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pagerank/XL/graph500-27/max_mem_30G_workers_16/wall_time.png'
set title "pagerank / graph500-27 (XL) — max_mem_30G workers_16\nmedian=418.515s  mean=405.257s  std=30.226s  min=351.271s  max=420.355s  p90=420.228s  p95=420.291s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.145221]
set xrange [342.258367:429.367994]
set grid y
set key top right
set arrow from 418.514803,0 to 418.514803,0.145221 nohead lc rgb 'red' lw 2
set label 'median 418.515s' at 418.514803,0.145221 offset char 0,1 tc rgb 'red'
set arrow from 420.227899,0 to 420.227899,0.145221 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 420.228s' at 420.227899,0.145221 offset char 0,1 tc rgb 'orange'
set arrow from 420.291392,0 to 420.291392,0.145221 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 420.291s' at 420.291392,0.145221 offset char 0,1 tc rgb 'orange'
set arrow from 351.271475,0 to 351.271475,0.00446834 nohead lc rgb '#666666' lw 1
set arrow from 416.107284,0 to 416.107284,0.00446834 nohead lc rgb '#666666' lw 1
set arrow from 420.354885,0 to 420.354885,0.00446834 nohead lc rgb '#666666' lw 1
set arrow from 418.514803,0 to 418.514803,0.00446834 nohead lc rgb '#666666' lw 1
set arrow from 420.037420,0 to 420.037420,0.00446834 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pagerank/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pagerank/XL/graph500-27/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
