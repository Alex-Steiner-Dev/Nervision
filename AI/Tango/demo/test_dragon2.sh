python eval.py \
--obj_path data/source_meshes/dragon_face5000.obj \
--output_dir results/demo/dragon/fire2 \
--prompt "a fire dragon" \
--width 468 \
--background 'black' \
--init_r_and_s \
--init_roughness 0.7 \
--local_percentage 0.4 \
--symmetry \
--radius 2.0 \
--n_views 3 \
--material_random_pe_sigma  20 \
--material_random_pe_numfreq  256 \
--num_lgt_sgs 32 \
--n_normaugs 4 \
--n_augs 1  \
--frontview_std 2 \
--clipavg view \
--lr_decay 0.9 \
--mincrop 0.05 \
--maxcrop 0.2 \
--seed 150 \
--n_iter 1501 \
--learning_rate 0.0005 \
--frontview_center 1.96349 0.6283  \
--model_dir results/demo/dragon/fire2/iter1500.pth \
--render_gif
