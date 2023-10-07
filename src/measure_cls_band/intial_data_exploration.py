#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import sys
sys.path.append("..")
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xarray as xr
import tqdm
import utm

#xr.set_options(file_cache_maxsize=1)  # From 

from src.utils.polygon_mask import convex_polygon_mask
from src.utils.polygon import Polygon


# In[2]:


data_dir = Path("../data_tif")


# In[3]:


product_files = list(data_dir.glob("*.tif"))


# In[4]:


# Defined in UTM coords, assuming UTM zone is identical to imagery
# All of Scotland is in zone 30N so this should be fine

# Define patches in lat/lon coords, then automatically convert to UTM

patch_regions_lat_lon = {
    "northern_cairngorms": [
        (57.114301, -3.929280),
        (57.162312, -3.714076),
        (57.152050, -3.358325),
        (57.006323, -3.397939),
        (57.003096, -3.902027)
    ],
    "monadh_liath": [
        (57.236986, -4.106196),
        (57.105345, -3.974958),
        (57.019243, -4.283127),
        (57.130311, -4.418831)
    ],
    "southern_cairngorms": [
        (57.048123, -3.037983),
        (56.871318, -3.177420),
        (56.814204, -3.460366),
        (57.006297, -3.398833)
    ],
    "beinn_dearg": [
        (56.898559, -3.939602),
        (56.901257, -3.836427),
        (56.873347, -3.821048),
        (56.828310, -3.831984),
        (56.850181, -3.917934),
    ]
}

convert_all = lambda x: [utm.from_latlon(*coord)[:2] for coord in x]
patch_regions_utm = {k: convert_all(v) for k, v in patch_regions_lat_lon.items()}        

patch_regions = {k: Polygon(vertices=v[::-1]) for k, v in patch_regions_utm.items()}

example_dataset = xr.open_dataset(product_files[0], engine="rasterio").astype(np.int8)


# In[5]:


example_dataset


# In[6]:


patch_masks = {}

for name, poly in patch_regions.items():
    convex_polygon_mask(poly, example_dataset)
    patch_masks[name] = example_dataset["mask"].copy()


# In[7]:


def date_from_filename(name: str) -> np.datetime64:
    date_str = name.split("_")[1]
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    hour = date_str[9:11]
    minute = date_str[11:13]
    second = date_str[13:15]
    date_str_formatted = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    date = np.datetime64(date_str_formatted)
    return date

product_dates = [date_from_filename(str(x.name)) for x in product_files]


# In[14]:


x = 0

dataset = xr.open_dataset(product_files[x], engine="rasterio").astype(np.int8)
scene_mask = dataset["band_data"][0]

date = product_dates[x]

scene_mask


# In[15]:


from matplotlib.colors import ListedColormap, BoundaryNorm

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
        plt.close()
    else:
        plt.show()

plot_scl(scene_mask, date, patch_regions)


# In[10]:


scene_mask.plot.hist(bins=11)


# In[11]:


from dataclasses import dataclass

@dataclass
class SnowMeasurement:
    name: str
    date: np.datetime64
    null_px: int = 0
    snow_px: int = 0
    cloud_px: int = 0
    shadow_px: int = 0
    vegetation_px: int = 0
    not_vegetation_px: int = 0
    water_px: int = 0
    uncertain_px: int = 0

    def to_dict(self):
        return vars(self)


def measure_snow(dataset, polygon, date, name):
    print(f"{name}: {date}")
    #convex_polygon_mask(polygon, dataset)
    zoom = scene_mask.where(patch_masks[name] == True, drop=True)

    result = SnowMeasurement(name=name, date=date)
    result.null_px = ((zoom == 0) | (zoom == 1)).sum().item()
    result.snow_px = (zoom == 11).sum().item()
    result.cloud_px = ((zoom == 8) | (zoom == 9) | (zoom == 10)).sum().item()
    result.shadow_px = ((zoom == 2) | (zoom == 3)).sum().item()
    result.vegetation_px = (zoom == 4).sum().item()
    result.not_vegetation_px = (zoom == 5).sum().item()
    result.water_px = (zoom == 6).sum().item()
    result.uncertain_px = (zoom == 7).sum().item()
    
    zoom = zoom.fillna(0)
    filename = f"./../output/{name}_{date}.png"
    plot_scl(zoom, date, filename=filename, title=f"{name}: {date}")
    
    return result


#results = {}
#for name, polygon in patch_regions.items():
#    results[name] = measure_snow(dataset, polygon, date, name)
#
#results


# In[12]:


# In this cell we run the measurement on every piece of data collected so far, and collect all pixel type measurements into a big list

all_results = []

for i in tqdm.tqdm(range(len(product_files))):
    with xr.open_dataset(product_files[i], engine="rasterio") as dataset:
        dataset = dataset.astype(np.int8)
        
        scene_mask = dataset["band_data"][0]
        date = product_dates[i]
    
        plot_scl(scene_mask, date, patch_regions, filename=f"./../output/T30VVJ_{date}.png")
    
        #for name, polygon in patch_regions.items():
        #    all_results.append(measure_snow(dataset, polygon, date, name))
    
        #scene_mask.close()
        #dataset.close()
        #gc.collect()


# In[ ]:


df = pd.DataFrame(all_results)

df


# In[ ]:





# In[ ]:




