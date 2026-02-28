python launch.py --config configs/edit-inf.yaml \
--train \
--gpu 2 \
system.max_densify_percent=0.01 \
system.anchor_weight_init_g0=0.05 \
system.anchor_weight_init=0.1 \
system.anchor_weight_multiplier=1.3 \
system.seg_prompt="face" \
system.loss.lambda_anchor_color=0 \
system.loss.lambda_anchor_geo=50 \
system.loss.lambda_anchor_scale=50 \
system.loss.lambda_anchor_opacity=50 \
system.densify_from_iter=100 \
system.densify_until_iter=1501 \
system.densification_interval=100 \
data.source=/data/users/jaeyeonpark/dataset/3d-ovs/covered_desk \
system.gs_source=/data/users/jaeyeonpark/3dgs-trained/3d-ovs/covered_desk/point_cloud/iteration_30000/point_cloud.ply \
system.guidance.src_prompt="face" \
system.guidance.tgt_prompt="elf" \
system.prompt_processor.prompt="elf" \
system.guidance.self_attn_th=0.8 \
system.guidance.cross_attn_th=0.6 \
system.guidance.src_blend_th=0.5 \
system.guidance.tgt_blend_th=0.5 \
trainer.max_steps=800 \
system.per_editing_step=400 \
data.max_view_num=20 \
system.guidance.skip_first_ctrl=False
# max_view_num 원래는 96 씀!!