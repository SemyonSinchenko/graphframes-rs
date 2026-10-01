set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/gfrs/wcc/XL/graph500-26/max_mem_30G_workers_16/wall_time.png'
set title "wcc / graph500-26 (XL) — max_mem_30G workers_16\nmedian=212.808s  mean=213.145s  std=2.104s  min=210.970s  max=216.485s  p90=215.314s  p95=215.900s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:0.282409]
set xrange [207.181952:220.273815]
set grid y
set key top right
set arrow from 212.807721,0 to 212.807721,0.282409 nohead lc rgb 'red' lw 2
set label 'median 212.808s' at 212.807721,0.282409 offset char 0,1 tc rgb 'red'
set arrow from 215.314295,0 to 215.314295,0.282409 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 215.314s' at 215.314295,0.282409 offset char 0,1 tc rgb 'orange'
set arrow from 215.899785,0 to 215.899785,0.282409 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 215.900s' at 215.899785,0.282409 offset char 0,1 tc rgb 'orange'
set arrow from 213.557822,0 to 213.557822,0.0086895 nohead lc rgb '#666666' lw 1
set arrow from 216.485276,0 to 216.485276,0.0086895 nohead lc rgb '#666666' lw 1
set arrow from 212.807721,0 to 212.807721,0.0086895 nohead lc rgb '#666666' lw 1
set arrow from 211.905843,0 to 211.905843,0.0086895 nohead lc rgb '#666666' lw 1
set arrow from 210.970491,0 to 210.970491,0.0086895 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/gfrs/wcc/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/gfrs/wcc/XL/graph500-26/max_mem_30G_workers_16/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
