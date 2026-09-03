from matplotlib.colors import ListedColormap

# Colour palette
blue    = "#648fff"  # 2017
orange  = "#fe6100"  # 2019
red     = "#dc267f"  # 2021
rgb_red = [220, 38, 127]
purple  = "#785EF0"  # overall
yellow  = "#FFB000"

# 0 = land (brown), 1 = water (blue)
mask_cmap = ListedColormap([orange, blue])
edge_cmap = ListedColormap([red])
