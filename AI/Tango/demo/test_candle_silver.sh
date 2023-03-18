python eval.py  \
--obj_path data/source_meshes/candle.obj \
--output_dir results/demo/candle/silver \
--init_r_and_s \
--init_roughness 0.1 \
--max_delta_theta 0.32359 \
--max_delta_phi 0.32359 \
--prompt a candle made of silver \
--width 512 \
--local_percentage 0.3 \
--background 'black'  \
--radius 2 \
--n_views 1 \
--material_random_pe_numfreq 3 \
--material_random_pe_sigma 0.5 \
--num_lgt_sgs 64 \
--n_normaugs 4 \
--n_augs 1 \
--frontview_std 16 \
--clipavg view \
--lr_decay 0.7 \
--mincrop 0.05 \
--maxcrop 0.05 \
--seed 250 \
--n_iter 1501 \
--learning_rate 0.0005 \
--frontview_center 1.96349 0.6283 \
--model_dir results/demo/candle/silver/iter400.pth \
--render_gif
