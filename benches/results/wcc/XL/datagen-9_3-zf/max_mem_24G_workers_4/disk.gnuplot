set terminal pngcairo size 1400,700 enhanced font 'Sans,11'
set output '/home/ubuntu/nvm/results/wcc/XL/datagen-9_3-zf/max_mem_24G_workers_4/disk.png'
set title "wcc / datagen-9_3-zf (XL) — max_mem_24G workers_4 — disk usage"
set xlabel 'fraction of run (%)'
set ylabel 'disk consumed (GiB)'
set grid
set key top left
set yrange [0:*]
plot '/home/ubuntu/nvm/results/wcc/XL/datagen-9_3-zf/max_mem_24G_workers_4/disk.dat' using 1:3:4 with filledcurves lc rgb '#cccccc' title '95% CI', \
     '' using 1:2 with lines lw 2 lc rgb '#4682b4' title 'mean'
