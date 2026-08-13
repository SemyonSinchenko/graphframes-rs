set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/XS/cit-Patents/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / cit-Patents (XS) — max_mem_24G workers_4\nmedian=12.403s  mean=12.460s  std=0.292s  min=12.232s  max=12.945s  p90=12.761s  p95=12.853s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:2.19537]
set xrange [11.659753:13.517010]
set grid y
set key top right
set arrow from 12.403252,0 to 12.403252,2.19537 nohead lc rgb 'red' lw 2
set label 'median 12.403s' at 12.403252,2.19537 offset char 0,1 tc rgb 'red'
set arrow from 12.760848,0 to 12.760848,2.19537 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 12.761s' at 12.760848,2.19537 offset char 0,1 tc rgb 'orange'
set arrow from 12.852719,0 to 12.852719,2.19537 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 12.853s' at 12.852719,2.19537 offset char 0,1 tc rgb 'orange'
set arrow from 12.232173,0 to 12.232173,0.0675497 nohead lc rgb '#666666' lw 1
set arrow from 12.235633,0 to 12.235633,0.0675497 nohead lc rgb '#666666' lw 1
set arrow from 12.944591,0 to 12.944591,0.0675497 nohead lc rgb '#666666' lw 1
set arrow from 12.403252,0 to 12.403252,0.0675497 nohead lc rgb '#666666' lw 1
set arrow from 12.485235,0 to 12.485235,0.0675497 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/XS/cit-Patents/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
