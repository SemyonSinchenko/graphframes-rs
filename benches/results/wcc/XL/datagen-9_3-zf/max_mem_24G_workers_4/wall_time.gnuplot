set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/datagen-9_3-zf/max_mem_24G_workers_4/wall_time.png'
set title "wcc / datagen-9_3-zf (XL) — max_mem_24G workers_4\nmedian=3027.081s  mean=3035.799s  std=54.177s  min=2984.779s  max=3119.912s  p90=3093.018s  p95=3106.465s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.00924807]
set xrange [2851.469937:3253.221815]
set grid y
set key top right
set arrow from 3027.080724,0 to 3027.080724,0.00924807 nohead lc rgb 'red' lw 2
set label 'median 3027.081s' at 3027.080724,0.00924807 offset char 0,1 tc rgb 'red'
set arrow from 3093.017942,0 to 3093.017942,0.00924807 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 3093.018s' at 3093.017942,0.00924807 offset char 0,1 tc rgb 'orange'
set arrow from 3106.465147,0 to 3106.465147,0.00924807 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 3106.465s' at 3106.465147,0.00924807 offset char 0,1 tc rgb 'orange'
set arrow from 2994.547185,0 to 2994.547185,0.000284556 nohead lc rgb '#666666' lw 1
set arrow from 3027.080724,0 to 3027.080724,0.000284556 nohead lc rgb '#666666' lw 1
set arrow from 3119.912352,0 to 3119.912352,0.000284556 nohead lc rgb '#666666' lw 1
set arrow from 2984.779400,0 to 2984.779400,0.000284556 nohead lc rgb '#666666' lw 1
set arrow from 3052.676328,0 to 3052.676328,0.000284556 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/wcc/XL/datagen-9_3-zf/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/wcc/XL/datagen-9_3-zf/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
