set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/cdlp/XS/cit-Patents/max_mem_24G_workers_4/wall_time.png'
set title "cdlp / cit-Patents (XS) — max_mem_24G workers_4\nmedian=44.604s  mean=44.615s  std=0.361s  min=44.142s  max=45.049s  p90=44.978s  p95=45.014s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:1.10514]
set xrange [43.074860:46.116041]
set grid y
set key top right
set arrow from 44.603885,0 to 44.603885,1.10514 nohead lc rgb 'red' lw 2
set label 'median 44.604s' at 44.603885,1.10514 offset char 0,1 tc rgb 'red'
set arrow from 44.978318,0 to 44.978318,1.10514 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 44.978s' at 44.978318,1.10514 offset char 0,1 tc rgb 'orange'
set arrow from 45.013619,0 to 45.013619,1.10514 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 45.014s' at 45.013619,1.10514 offset char 0,1 tc rgb 'orange'
set arrow from 44.141981,0 to 44.141981,0.0340043 nohead lc rgb '#666666' lw 1
set arrow from 44.872417,0 to 44.872417,0.0340043 nohead lc rgb '#666666' lw 1
set arrow from 44.603885,0 to 44.603885,0.0340043 nohead lc rgb '#666666' lw 1
set arrow from 45.048919,0 to 45.048919,0.0340043 nohead lc rgb '#666666' lw 1
set arrow from 44.407103,0 to 44.407103,0.0340043 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/cdlp/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/cdlp/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
