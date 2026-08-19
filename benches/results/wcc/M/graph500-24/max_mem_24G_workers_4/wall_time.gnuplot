set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/M/graph500-24/max_mem_24G_workers_4/wall_time.png'
set title "wcc / graph500-24 (M) — max_mem_24G workers_4\nmedian=319.919s  mean=320.505s  std=2.349s  min=318.332s  max=324.509s  p90=322.783s  p95=323.646s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.687936]
set xrange [316.904801:325.935853]
set grid y
set key top right
set arrow from 319.918848,0 to 319.918848,0.687936 nohead lc rgb 'red' lw 2
set label 'median 319.919s' at 319.918848,0.687936 offset char 0,1 tc rgb 'red'
set arrow from 322.783029,0 to 322.783029,0.687936 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 322.783s' at 322.783029,0.687936 offset char 0,1 tc rgb 'orange'
set arrow from 323.645946,0 to 323.645946,0.687936 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 323.646s' at 323.645946,0.687936 offset char 0,1 tc rgb 'orange'
set arrow from 320.194278,0 to 320.194278,0.0211673 nohead lc rgb '#666666' lw 1
set arrow from 324.508863,0 to 324.508863,0.0211673 nohead lc rgb '#666666' lw 1
set arrow from 319.572044,0 to 319.572044,0.0211673 nohead lc rgb '#666666' lw 1
set arrow from 318.331790,0 to 318.331790,0.0211673 nohead lc rgb '#666666' lw 1
set arrow from 319.918848,0 to 319.918848,0.0211673 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/M/graph500-24/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
