## visualize retarget

python visualize_dexonomy_grasp.py \
    --robots shadow_no_wrist omni \
    --retargeting_type vector \
    --two_optimizers \
    --second_optimizer_type FINGERTIP \
    --data_idx 0

# store retarget

python store_dexonomy_retarget.py   --robots shadow_no_wrist omni   --retargeting-type VECTOR 

python store_dexonomy_retarget.py   --robots shadow_no_wrist omni   --retargeting-type VECTOR --two-optimizers --second-optimizer-type FINGERTIP --visualize --data-id-start 0 --data-id-end 1