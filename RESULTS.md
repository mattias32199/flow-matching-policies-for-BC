# Results

Status: setup and validation in progress; no reproduction claims yet.

## Planned comparisons

- Mystery 1: block-double with fixed 10,000 episodes; track training/validation flow loss, action MSE, and success over training. Compare block-single with independently sampled 10,000 versus 50,000 episodes over four seeds.
- Mystery 4: block-double streaming demonstrations; compare handcrafted observations, standardized observations, and standardized observations with object/gripper xyz scaled by 0.1 or 10.

Reference defaults: eight 4096-wide residual MLP layers, action chunks of 25, batch size 1024, learning rate 1e-4, 10 flow steps. Blog-specific settings beyond the public reference commands remain unverified. Finite streaming runs approximate the infinite-data setting.
