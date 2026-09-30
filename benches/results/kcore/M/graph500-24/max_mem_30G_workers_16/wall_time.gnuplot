set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/kcore/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "kcore / graph500-24 (M) — max_mem_30G workers_16\nmedian=287.823s  mean=288.920s  std=3.015s  min=286.419s  max=293.401s  p90=292.253s  p95=292.827s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.14552]
set xrange [277.155019:302.664989]
set grid y
set key top right
set arrow from 287.823061,0 to 287.823061,0.14552 nohead lc rgb 'red' lw 2
set label 'median 287.823s' at 287.823061,0.14552 offset char 0,1 tc rgb 'red'
set arrow from 292.252795,0 to 292.252795,0.14552 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 292.253s' at 292.252795,0.14552 offset char 0,1 tc rgb 'orange'
set arrow from 292.826761,0 to 292.826761,0.14552 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 292.827s' at 292.826761,0.14552 offset char 0,1 tc rgb 'orange'
set arrow from 287.823061,0 to 287.823061,0.00447752 nohead lc rgb '#666666' lw 1
set arrow from 286.425199,0 to 286.425199,0.00447752 nohead lc rgb '#666666' lw 1
set arrow from 286.419281,0 to 286.419281,0.00447752 nohead lc rgb '#666666' lw 1
set arrow from 290.530898,0 to 290.530898,0.00447752 nohead lc rgb '#666666' lw 1
set arrow from 293.400727,0 to 293.400727,0.00447752 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/kcore/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/kcore/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
