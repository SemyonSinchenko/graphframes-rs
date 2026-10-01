set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/cdlp/M/graph500-24/max_mem_30G_workers_16/wall_time.png'
set title "cdlp / graph500-24 (M) — max_mem_30G workers_16\nmedian=250.230s  mean=251.523s  std=4.177s  min=247.173s  max=257.958s  p90=255.999s  p95=256.978s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.130497]
set xrange [238.299668:266.831420]
set grid y
set key top right
set arrow from 250.230070,0 to 250.230070,0.130497 nohead lc rgb 'red' lw 2
set label 'median 250.230s' at 250.230070,0.130497 offset char 0,1 tc rgb 'red'
set arrow from 255.999035,0 to 255.999035,0.130497 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 255.999s' at 255.999035,0.130497 offset char 0,1 tc rgb 'orange'
set arrow from 256.978344,0 to 256.978344,0.130497 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 256.978s' at 256.978344,0.130497 offset char 0,1 tc rgb 'orange'
set arrow from 250.230070,0 to 250.230070,0.00401529 nohead lc rgb '#666666' lw 1
set arrow from 253.061107,0 to 253.061107,0.00401529 nohead lc rgb '#666666' lw 1
set arrow from 257.957653,0 to 257.957653,0.00401529 nohead lc rgb '#666666' lw 1
set arrow from 249.191731,0 to 249.191731,0.00401529 nohead lc rgb '#666666' lw 1
set arrow from 247.173435,0 to 247.173435,0.00401529 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/cdlp/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/cdlp/M/graph500-24/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
