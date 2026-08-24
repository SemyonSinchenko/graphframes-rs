set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/cdlp/L/graph500-25/max_mem_24G_workers_4/wall_time.png'
set title "cdlp / graph500-25 (L) — max_mem_24G workers_4\nmedian=1714.938s  mean=1710.890s  std=17.682s  min=1683.916s  max=1729.697s  p90=1726.398s  p95=1728.048s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0294108]
set xrange [1644.932245:1768.681089]
set grid y
set key top right
set arrow from 1714.938128,0 to 1714.938128,0.0294108 nohead lc rgb 'red' lw 2
set label 'median 1714.938s' at 1714.938128,0.0294108 offset char 0,1 tc rgb 'red'
set arrow from 1726.398101,0 to 1726.398101,0.0294108 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 1726.398s' at 1726.398101,0.0294108 offset char 0,1 tc rgb 'orange'
set arrow from 1728.047636,0 to 1728.047636,0.0294108 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 1728.048s' at 1728.047636,0.0294108 offset char 0,1 tc rgb 'orange'
set arrow from 1729.697171,0 to 1729.697171,0.000904949 nohead lc rgb '#666666' lw 1
set arrow from 1704.450688,0 to 1704.450688,0.000904949 nohead lc rgb '#666666' lw 1
set arrow from 1714.938128,0 to 1714.938128,0.000904949 nohead lc rgb '#666666' lw 1
set arrow from 1721.449494,0 to 1721.449494,0.000904949 nohead lc rgb '#666666' lw 1
set arrow from 1683.916162,0 to 1683.916162,0.000904949 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/cdlp/L/graph500-25/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/cdlp/L/graph500-25/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
