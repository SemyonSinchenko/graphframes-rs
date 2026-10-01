set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/kcore/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "kcore / cit-Patents (XS) — max_mem_30G workers_16\nmedian=11.922s  mean=11.918s  std=0.041s  min=11.857s  max=11.972s  p90=11.953s  p95=11.962s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:34.289]
set xrange [11.829170:11.999886]
set grid y
set key top right
set arrow from 11.922367,0 to 11.922367,34.289 nohead lc rgb 'red' lw 2
set label 'median 11.922s' at 11.922367,34.289 offset char 0,1 tc rgb 'red'
set arrow from 11.952988,0 to 11.952988,34.289 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 11.953s' at 11.952988,34.289 offset char 0,1 tc rgb 'orange'
set arrow from 11.962303,0 to 11.962303,34.289 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 11.962s' at 11.962303,34.289 offset char 0,1 tc rgb 'orange'
set arrow from 11.857439,0 to 11.857439,1.05505 nohead lc rgb '#666666' lw 1
set arrow from 11.925042,0 to 11.925042,1.05505 nohead lc rgb '#666666' lw 1
set arrow from 11.912716,0 to 11.912716,1.05505 nohead lc rgb '#666666' lw 1
set arrow from 11.922367,0 to 11.922367,1.05505 nohead lc rgb '#666666' lw 1
set arrow from 11.971618,0 to 11.971618,1.05505 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/kcore/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/kcore/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
