set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/sp/XL/graph500-26/max_mem_24G_workers_4/wall_time.png'
set title "sp / graph500-26 (XL) — max_mem_24G workers_4\nmedian=634.449s  mean=634.004s  std=12.146s  min=619.027s  max=650.595s  p90=646.203s  p95=648.399s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.0354766]
set xrange [588.568698:681.053610]
set grid y
set key top right
set arrow from 634.448723,0 to 634.448723,0.0354766 nohead lc rgb 'red' lw 2
set label 'median 634.449s' at 634.448723,0.0354766 offset char 0,1 tc rgb 'red'
set arrow from 646.203332,0 to 646.203332,0.0354766 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 646.203s' at 646.203332,0.0354766 offset char 0,1 tc rgb 'orange'
set arrow from 648.399259,0 to 648.399259,0.0354766 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 648.399s' at 648.399259,0.0354766 offset char 0,1 tc rgb 'orange'
set arrow from 626.334260,0 to 626.334260,0.00109159 nohead lc rgb '#666666' lw 1
set arrow from 634.448723,0 to 634.448723,0.00109159 nohead lc rgb '#666666' lw 1
set arrow from 639.615553,0 to 639.615553,0.00109159 nohead lc rgb '#666666' lw 1
set arrow from 650.595185,0 to 650.595185,0.00109159 nohead lc rgb '#666666' lw 1
set arrow from 619.027123,0 to 619.027123,0.00109159 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/sp/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/sp/XL/graph500-26/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
