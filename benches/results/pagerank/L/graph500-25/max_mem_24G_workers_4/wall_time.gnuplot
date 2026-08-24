set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/pagerank/L/graph500-25/max_mem_24G_workers_4/wall_time.png'
set title "pagerank / graph500-25 (L) — max_mem_24G workers_4\nmedian=211.402s  mean=211.614s  std=2.849s  min=208.115s  max=216.069s  p90=214.208s  p95=215.138s  runs=5"
set xlabel 'wall time (s)'
set ylabel 'probability density (1/s)'
set tmargin 5
set yrange [0:1.18044]
set xrange [207.313452:216.870713]
set grid y
set key top right
set arrow from 211.402147,0 to 211.402147,1.18044 nohead lc rgb 'red' lw 2
set label 'median 211.402s' at 211.402147,1.18044 offset char 0,1 tc rgb 'red'
set arrow from 214.207644,0 to 214.207644,1.18044 nohead lc rgb 'orange' lw 1 dt 2
set label 'p90 214.208s' at 214.207644,1.18044 offset char 0,1 tc rgb 'orange'
set arrow from 215.138184,0 to 215.138184,1.18044 nohead lc rgb 'orange' lw 1 dt 3
set label 'p95 215.138s' at 215.138184,1.18044 offset char 0,1 tc rgb 'orange'
set arrow from 211.416021,0 to 211.416021,0.0363214 nohead lc rgb '#666666' lw 1
set arrow from 208.115440,0 to 208.115440,0.0363214 nohead lc rgb '#666666' lw 1
set arrow from 216.068725,0 to 216.068725,0.0363214 nohead lc rgb '#666666' lw 1
set arrow from 211.066317,0 to 211.066317,0.0363214 nohead lc rgb '#666666' lw 1
set arrow from 211.402147,0 to 211.402147,0.0363214 nohead lc rgb '#666666' lw 1
plot '/home/ubuntu/nvm/results/pagerank/L/graph500-25/max_mem_24G_workers_4/wall_time.dat' using 1:2 with filledcurves y=0 lc rgb '#4682b4' fs transparent solid 0.25 title 'kernel density', \
     '/home/ubuntu/nvm/results/pagerank/L/graph500-25/max_mem_24G_workers_4/wall_time.dat' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'KDE (runs)'
