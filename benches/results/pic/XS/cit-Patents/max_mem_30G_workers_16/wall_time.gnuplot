set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/pic/XS/cit-Patents/max_mem_30G_workers_16/wall_time.png'
set title "pic / cit-Patents (XS) — max_mem_30G workers_16\nmedian=17.482s  mean=17.525s  std=0.130s  min=17.400s  max=17.723s  p90=17.666s  p95=17.695s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:3.82808]
set xrange [17.070381:18.052887]
set grid y
set key top right
set arrow from 17.482024,0 to 17.482024,3.82808 nohead lc rgb 'red' lw 2
set label 'median 17.482s' at 17.482024,3.82808 offset char 0,1 tc rgb 'red'
set arrow from 17.666385,0 to 17.666385,3.82808 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 17.666s' at 17.666385,3.82808 offset char 0,1 tc rgb 'orange'
set arrow from 17.694832,0 to 17.694832,3.82808 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 17.695s' at 17.694832,3.82808 offset char 0,1 tc rgb 'orange'
set arrow from 17.581042,0 to 17.581042,0.117787 nohead lc rgb '#666666' lw 1
set arrow from 17.482024,0 to 17.482024,0.117787 nohead lc rgb '#666666' lw 1
set arrow from 17.399989,0 to 17.399989,0.117787 nohead lc rgb '#666666' lw 1
set arrow from 17.437318,0 to 17.437318,0.117787 nohead lc rgb '#666666' lw 1
set arrow from 17.723280,0 to 17.723280,0.117787 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/pic/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/pic/XS/cit-Patents/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
