set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/ldbd/wcc/M/graph500-24/icebug_mem_12G_threads_4/wall_time.png'
set title "icebug wcc / graph500-24 (M) — mem_12G_threads_4\nmedian=199.797s  mean=199.481s  std=2.046s  min=197.086s  max=201.935s  p90=201.499s  p95=201.717s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.189068]
set xrange [190.797692:208.222831]
set grid y
set key top right
set arrow from 199.796942,0 to 199.796942,0.189068 nohead lc rgb 'red' lw 2
set label 'median 199.797s' at 199.796942,0.189068 offset char 0,1 tc rgb 'red'
set arrow from 201.499062,0 to 201.499062,0.189068 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 201.499s' at 201.499062,0.189068 offset char 0,1 tc rgb 'orange'
set arrow from 201.716843,0 to 201.716843,0.189068 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 201.717s' at 201.716843,0.189068 offset char 0,1 tc rgb 'orange'
set arrow from 197.740539,0 to 197.740539,0.00581748 nohead lc rgb '#666666' lw 1
set arrow from 201.934625,0 to 201.934625,0.00581748 nohead lc rgb '#666666' lw 1
set arrow from 200.845719,0 to 200.845719,0.00581748 nohead lc rgb '#666666' lw 1
set arrow from 199.796942,0 to 199.796942,0.00581748 nohead lc rgb '#666666' lw 1
set arrow from 197.085898,0 to 197.085898,0.00581748 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/ldbd/wcc/M/graph500-24/icebug_mem_12G_threads_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/ldbd/wcc/M/graph500-24/icebug_mem_12G_threads_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
