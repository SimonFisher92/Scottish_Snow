from pathlib import Path
from dataclasses import dataclass
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xarray as xr
import tqdm
import rasterio as rio
import rioxarray

from src.measure_cls_band.polygon_mask import convex_polygon_mask
from src.measure_cls_band.plotting import plot_scl

logger = logging.getLogger(__name__)
logging.getLogger("rasterio").setLevel(logging.WARNING)


@dataclass
class SnowMeasurement:
    """Data class to store pixel counts from a ROI
    """
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


def measure_snow(dataset, roi_masks, date, name, args):
    """Count different SCL classes in a particular ROI of a dataset
    """
    zoom = dataset["SCL"].where(roi_masks[name] == True, drop=True)

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
    filename = str(Path(args.output_dir) / f"{name}_{date}.png")
    plot_scl(zoom, date, filename=filename, title=f"{name}: {date}")
    
    return result


def precompute_patch_masks(rois, data_list, data_to_jp2s) -> dict[str, xr.DataArray]:
    """In this function we compute boolean mask arrays for each ROI
    """

    # First find an arbitrary dataset to get coordinate arrays from
    # This assumes that all data has identical geolocation
    i = 0
    while True:
        try:
            example_jp2_path = data_to_jp2s[data_list[i]][0]
            break
        except IndexError:
            i += 1

    example_dataset = rioxarray.open_rasterio(example_jp2_path).squeeze(drop=True).to_dataset(name="SCL")

    roi_masks = {}
    
    for name, poly in rois.items():
        convex_polygon_mask(poly, example_dataset)
        roi_masks[name] = example_dataset["mask"].copy()

    return roi_masks


def date_from_filename(name: str) -> np.datetime64:
    """Parse a .jp2 filename to extract dates
    """
    date_str = name.split("_")[1]
    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]
    hour = date_str[9:11]
    minute = date_str[11:13]
    second = date_str[13:15]
    date_str_formatted = f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    date = np.datetime64(date_str_formatted)

    logger.debug(f"Extracted date: {date} from name: {name}")

    return date


def extract_time_series(rois, args) -> pd.DataFrame:
    """High level function to extract time series of pixel type counts for each ROI
    """
    data_list = list(Path(args.data_dir).glob("*.SAFE"))  # List of paths to .SAFE dirs
    data_to_jp2s = {safe_file: list(safe_file.glob("**/*.jp2")) for safe_file in data_list}

    #for k, v in data_to_jp2s.items():
    #    assert len(v) == 1, f"{k} has unexpected amount of jp2 data: {v}"

    roi_masks = precompute_patch_masks(rois, data_list, data_to_jp2s)

    all_results = []
    
    for i in tqdm.tqdm(range(len(data_list))):
        try:
            jp2_path = data_to_jp2s[data_list[i]][0]
        except IndexError as e:
            logger.warning(e)
            continue
        date = date_from_filename(jp2_path.stem)
        dataset = rioxarray.open_rasterio(jp2_path).squeeze(drop=True).to_dataset(name="SCL")

        logger.debug(f"Opened dataset: {dataset}")
            
        plot_scl(dataset["SCL"], date, rois, filename=str(Path(args.output_dir) / f"T30VVJ_{date}.png"))

        for name, polygon in rois.items():
            all_results.append(measure_snow(dataset, roi_masks, date, name, args))

        dataset.close()

    return pd.DataFrame(all_results)
