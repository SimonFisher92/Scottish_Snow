import logging

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


def plot_scl(data_array, date, patch_regions = None, filename = None, title = None):

    # Create a custom colormap with a unique color for each class
    colors = ['#000000',   # 0: NO_DATA (Black)
          '#FF0000',   # 1: SATURATED_OR_DEFECTIVE (Red)
          '#404040',   # 2: CAST_SHADOWS (Dark Grey)
          '#8B4513',   # 3: CLOUD_SHADOWS (Brown)
          '#008000',   # 4: VEGETATION (Green)
          '#FFFF00',   # 5: NOT_VEGETATED (Yellow)
          '#0000FF',   # 6: WATER (Blue)
          '#FFA500',   # 7: UNCLASSIFIED (Orange)
          '#D3D3D3',   # 8: CLOUD_MEDIUM_PROBABILITY (Light Grey)
          '#FFFFFF',   # 9: CLOUD_HIGH_PROBABILITY (White)
          '#ADD8E6',   # 10: THIN_CIRRUS (Light Blue)
          '#FFC0CB']   # 11: SNOW or ICE (Pink)
    
    cmap = ListedColormap(colors)
    
    # Define the boundaries for the colormap (0 to 11)
    bounds = list(range(13))
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot the DataArray using the custom colormap and norm
    #im = data_array.plot.imshow(ax=ax, cmap=cmap, norm=norm)
    im = ax.imshow(data_array, cmap=cmap, norm=norm, interpolation="nearest", extent=[data_array.x.min(), data_array.x.max(), data_array.y.min(), data_array.y.max()])

    if patch_regions is not None:
        for name, p in patch_regions.items():
            polygon = matplotlib.patches.Polygon(p.vertices, closed=True, edgecolor='black', linewidth=1, facecolor='none')
            ax.add_patch(polygon)
            ax.text(p.vertices[np.argmax([x[0] for x in p.vertices])][0], p.vertices[np.argmax([x[0] for x in p.vertices])][1], name, bbox=dict(boxstyle='larrow', facecolor='white', alpha=0.6))
    
    # Set axis labels and title
    ax.set_xlabel('X [UTM]')
    ax.set_ylabel('Y [UTM]')
    if title is None:
        ax.set_title(f'Sentinel 2 Scene Classification: {date}')
    else:
        ax.set_title(title)
    
    # Create a colorbar with class labels
    cbar = plt.colorbar(im, ax=ax, cmap=cmap, boundaries=bounds, ticks=[x + 0.5 for x in bounds])
    cbar.ax.set_yticklabels([
        'NO_DATA', 'SATURATED_OR_DEFECTIVE', 'CAST_SHADOWS', 'CLOUD_SHADOWS',
        'VEGETATION', 'NOT_VEGETATED', 'WATER', 'UNCLASSIFIED',
        'CLOUD_MEDIUM_PROBABILITY', 'CLOUD_HIGH_PROBABILITY',
        'THIN_CIRRUS', 'SNOW or ICE', ""
    ])
    
    if filename:
        plt.savefig(filename)
        logger.info(f"Saved plot to {filename}")
        plt.close()
    else:
        plt.show()
