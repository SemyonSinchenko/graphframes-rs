set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/wcc/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "wcc / cit-Patents (XS) — max_mem_30G workers_16\nmedian=4.709s  mean=4.711s  std=0.006s  min=4.705s  max=4.718s  p90=4.717s  p95=4.718s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:73.8801]
set xrange [4.687952:4.735139]
set grid y
set key top right
set arrow from 4.708698,0 to 4.708698,73.8801 nohead lc rgb 'red' lw 2
set label 'median 4.709s' at 4.708698,73.8801 offset char 0,1 tc rgb 'red'
set arrow from 4.717041,0 to 4.717041,73.8801 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 4.717s' at 4.717041,73.8801 offset char 0,1 tc rgb 'orange'
set arrow from 4.717555,0 to 4.717555,73.8801 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 4.718s' at 4.717555,73.8801 offset char 0,1 tc rgb 'orange'
set arrow from 4.715497,0 to 4.715497,2.27324 nohead lc rgb '#666666' lw 1
set arrow from 4.718070,0 to 4.718070,2.27324 nohead lc rgb '#666666' lw 1
set arrow from 4.707514,0 to 4.707514,2.27324 nohead lc rgb '#666666' lw 1
set arrow from 4.708698,0 to 4.708698,2.27324 nohead lc rgb '#666666' lw 1
set arrow from 4.705021,0 to 4.705021,2.27324 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/wcc/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/wcc/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
