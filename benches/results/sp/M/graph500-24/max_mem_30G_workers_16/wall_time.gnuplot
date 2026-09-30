set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/sp/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "sp / graph500-24 (M) — max_mem_30G workers_16\nmedian=6.681s  mean=6.720s  std=0.092s  min=6.644s  max=6.873s  p90=6.817s  p95=6.845s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:8.30759]
set xrange [6.494973:7.021843]
set grid y
set key top right
set arrow from 6.681236,0 to 6.681236,8.30759 nohead lc rgb 'red' lw 2
set label 'median 6.681s' at 6.681236,8.30759 offset char 0,1 tc rgb 'red'
set arrow from 6.816809,0 to 6.816809,8.30759 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 6.817s' at 6.816809,8.30759 offset char 0,1 tc rgb 'orange'
set arrow from 6.844950,0 to 6.844950,8.30759 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 6.845s' at 6.844950,8.30759 offset char 0,1 tc rgb 'orange'
set arrow from 6.667523,0 to 6.667523,0.255618 nohead lc rgb '#666666' lw 1
set arrow from 6.643725,0 to 6.643725,0.255618 nohead lc rgb '#666666' lw 1
set arrow from 6.732386,0 to 6.732386,0.255618 nohead lc rgb '#666666' lw 1
set arrow from 6.873091,0 to 6.873091,0.255618 nohead lc rgb '#666666' lw 1
set arrow from 6.681236,0 to 6.681236,0.255618 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/sp/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/sp/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
