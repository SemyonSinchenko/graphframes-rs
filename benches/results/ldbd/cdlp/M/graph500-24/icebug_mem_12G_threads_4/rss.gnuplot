set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/ldbd/cdlp/M/graph500-24/icebug_mem_12G_threads_4/rss.png'
set title "icebug cdlp / graph500-24 (M) — mem_12G_threads_4 — RSS"
set xlabel 'fraction of run (%)'
set ylabel 'RSS (GiB)'
set grid
set key top left
set yrange [0:*]
plot '/home/ubuntu/nvm/results/ldbd/cdlp/M/graph500-24/icebug_mem_12G_threads_4/rss.dat' using 1:3:4 with filledcurves lc rgb '#cccccc' title '95% CI', \
     '' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'mean'
