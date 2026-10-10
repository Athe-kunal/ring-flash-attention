.PHONY: visualizer

# Trace the varlen LSE Triton kernels and serve the TileLens UI.
visualizer:
	uv run python ring_flash_attn_video/visualize.py
