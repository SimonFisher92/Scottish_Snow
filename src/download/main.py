import datetime
import logging
from dateutil import parser
from pathlib import Path

import pandas as pd
from sentinelhub import DataCollection, SentinelHubCatalog, SentinelHubRequest, BBox, bbox_to_dimensions, CRS, \
    MimeType
import matplotlib
import numpy as np
import xarray as xr
from tqdm import tqdm

from src.download.utils import geojson_to_bbox, get_config, bbox_to_grid_coords, dataset_filename
from src.download.visualisation import plot_ds

logger = logging.getLogger()

# Basic parameters for the script
OUTDIR = Path("output/downloads/data")
VISUALISE_DATA = True

# Globals for the date interval which we get data from each year
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
START_MONTH = 5
START_DAY = 1
END_MONTH = 9
END_DAY = 30

# Data resolution which we download
RESOLUTION = 10

# Short version of code for debugging
DEV_MODE = False


def download_dem_data(geojson_path):
    evalscript_dem = """
    //VERSION=3
    function setup() {
      return {
        input: ["DEM"],
        output:{
          id: "default",
          bands: 1,
          sampleType: SampleType.FLOAT32
        }
      }
    }

    function evaluatePixel(sample) {
      return [sample.DEM]
    }
    """

    data_collection = DataCollection.DEM.define_from(name="dem", service_url="https://sh.dataspace.copernicus.eu")

    dem_data = get_sentinelhub_data(evalscript_dem, geojson_path, data_collection, ("2020-06-12", "2020-06-13")).get_data()

    logger.info("Downloaded DEM data")

    return dem_data[0]


def add_slope_information(dem_data: np.array) -> np.array:
    assert len(dem_data.shape) == 2

    spacing = RESOLUTION
    grad_lat, grad_lon = np.gradient(dem_data, spacing)

    # TODO: check this
    # I think that the gradient w.r.t latitude is pointing the wrong way, since in image space as "i" increases
    # then latitude descreases (because each image is orientated with north at the top).
    grad_lat *= -1

    result = np.stack([dem_data, grad_lon, grad_lat], axis=-1)
    return result


def get_sentinelhub_data(evalscript: str, geojson_path: Path, data_collection: DataCollection, time_interval: tuple):

    snowpatch_aoi = geojson_to_bbox(geojson_path)
    # lon, lat, lon, lat for BL and TR corners of box
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)
    # (number of lon measurement points, number of lat measurement points)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=RESOLUTION)

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=data_collection,
                time_interval=time_interval,
                # Arbitrary dates, elevation is constant on our time scales ;)
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=aoi_bbox,
        size=aoi_size,
        config=get_config(),
    )
    return request


def search_sentinelhub_catalog(data_collection, geojson_path: Path, time_interval: tuple[datetime.datetime, datetime.datetime]) -> list:
    # Search catalog for actual data products (tiles) which intersect this geometry and time interval

    catalog = SentinelHubCatalog(config=get_config())
    snowpatch_aoi = geojson_to_bbox(geojson_path)
    # lon, lat, lon, lat for BL and TR corners of box
    aoi_bbox = BBox(bbox=snowpatch_aoi, crs=CRS.WGS84)

    search_iterator = catalog.search(
        data_collection,
        bbox=aoi_bbox,
        time=time_interval,
        fields={"include": ["id", "properties.datetime"], "exclude": []},
    )

    # These objects are all data products found in the time interval, and overlapping with our aoi
    # Note that for the same date, we can still get multiple returns corresponding to different tiles
    results = list(search_iterator)

    return results


def extract_unique_dates_from_catalog_results(catalog_results) -> list[datetime.date]:
    # Extract unique days from the 'datetime' field
    unique_days = set()
    for item in catalog_results:
        datetime_str = item['properties']['datetime']
        date_object = parser.parse(datetime_str).date()
        unique_days.add(date_object)

    # Convert the set to a sorted list
    sorted_unique_days = sorted(list(unique_days))

    return sorted_unique_days


def l1c_evalscript():
    evalscript_all_bands = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"],
                    units: "DN"
                }],
                output: {
                    bands: 13,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B10,
                    sample.B11,
                    sample.B12];
        }
    """
    return evalscript_all_bands


def l2a_evalscript():
    evalscript_cls = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["SCL", "SNW", "CLD"],
                    units: "DN"
                }],
                output: {
                    bands: 3,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.SCL,
                    sample.SNW,
                    sample.CLD];
        }
    """
    return evalscript_cls


def download_l1c_data(year: int, geojson: Path) -> dict[datetime.date, np.array]:
    logger.info(f"Downloading L1C band data for year {year}")

    catalog_results = search_sentinelhub_catalog(
        data_collection=DataCollection.SENTINEL2_L1C,
        geojson_path=geojson,
        time_interval=(datetime.datetime(year=year, month=START_MONTH, day=START_DAY),
                       datetime.datetime(year=year, month=END_MONTH, day=END_DAY))
    )

    flyover_dates = extract_unique_dates_from_catalog_results(catalog_results)

    results = {}

    for date in tqdm(flyover_dates, desc=f'Getting data from {geojson.stem} in {year}'):

        # The time bounds are inclusive
        time_interval = (date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))

        data_collection = DataCollection.SENTINEL2_L1C.define_from(
            name="s2c", service_url="https://sh.dataspace.copernicus.eu"
        )

        # Note: SentinelHub will always return data, even if there was no flyover that day
        # Presumably this is the previous flyover to the requested day
        # This is why we also need to search the catalog first, so we only save data on days where there was a flyover
        l1c_data_point = get_sentinelhub_data(l1c_evalscript(), geojson, data_collection, time_interval).get_data()
        results[date] = l1c_data_point[0]

    return results


def download_l2a_data(year: int, geojson: Path) -> dict[datetime.date, np.array]:
    logger.info(f"Downloading L2A band data for year {year}")

    catalog_results = search_sentinelhub_catalog(
        data_collection=DataCollection.SENTINEL2_L2A,
        geojson_path=geojson,
        time_interval=(datetime.datetime(year=year, month=START_MONTH, day=START_DAY),
                       datetime.datetime(year=year, month=END_MONTH, day=END_DAY))
    )

    flyover_dates = extract_unique_dates_from_catalog_results(catalog_results)

    results = {}

    for date in tqdm(flyover_dates, desc=f'Getting data from {geojson.stem} in {year}'):
        # The time bounds are inclusive
        time_interval = (date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))

        data_collection = DataCollection.SENTINEL2_L2A.define_from(
            name="s2a", service_url="https://sh.dataspace.copernicus.eu"
        )

        l2a_data_point = get_sentinelhub_data(l2a_evalscript(), geojson, data_collection, time_interval).get_data()
        results[date] = l2a_data_point[0]

    return results


def construct_xarray_dataset(geojson_path, dem_data, l1c_data, l2a_data, dates) -> xr.Dataset:
    """Construct xarray dataset from all our data. This dataset contains all data for a single year, for a single patch
    """
    assert len(dem_data.shape) == 3
    assert len(l1c_data.shape) == 4
    assert len(l2a_data.shape) == 4
    assert l1c_data.shape[:3] == l2a_data.shape[:3]
    assert len(dates) == l2a_data.shape[0]
    assert len(dates) == l1c_data.shape[0]

    # First create coordinate arrays
    snowpatch_aoi = geojson_to_bbox(geojson_path)
    aoi_bbox = BBox(bbox=snowpatch_aoi,
                    crs=CRS.WGS84)  # CRS is a standard which specifies a reference ellipsoid for earth
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=RESOLUTION)
    lon_ccs, lat_ccs = bbox_to_grid_coords(aoi_bbox, aoi_size)

    l1c_channel_labels = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    l2a_channel_labels = ["SCL", "SNW", "CLD"]
    dem_channel_labels = ["Elevation", "Gradient_lon", "Gradient_lat"]

    data_vars = {l1c_channel_labels[i]: (["t", "x", "y"], l1c_data[..., i]) for i in range(13)}
    data_vars.update({l2a_channel_labels[i]: (["t", "x", "y"], l2a_data[..., i]) for i in range(3)})
    data_vars.update({dem_channel_labels[i]: (["x", "y"], dem_data[..., i]) for i in range(3)})

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"lon": (["x", "y"], lon_ccs),
                "lat": (["x", "y"], lat_ccs),
                "date": (["t"], pd.to_datetime(dates))},  # Dates must be datetime64 type to save ds to disk
        attrs={"description": "Sentinel L1C, L2A, and DEM data",
               "patchname": geojson_path.stem}
    )

    logger.debug(f"Constructed xarray dataset:\n{ds}")

    return ds


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created {OUTDIR.resolve()}")

    geojsons = list(Path(__file__).parent.parent.parent.glob('output/geojson_bboxes/*.geojson'))
    assert len(geojsons) > 0
    logger.info(f"Found geojsons:{[x.stem for x in geojsons]}")

    # set to true for debugging etc, setting to false will get all data
    if DEV_MODE:  # lightweight mode for development
        logger.info('You are in developer mode')
        global YEARS
        YEARS = [2020]
        global END_MONTH
        END_MONTH = 5
        geojsons = geojsons[:2]
    else:
        logger.info('You are in full mode, downloading all data, this will take several hours, set and forget')

    for geojson in geojsons:
        logger.info(f"Working on {geojson.stem}")
        dem_data = download_dem_data(geojson)  # Numpy array (height, width, 1)
        dem_data = add_slope_information(dem_data)  # Numpy array (height, width, 3)

        for year in YEARS:

            if (OUTDIR / dataset_filename(geojson, year)).exists():
                logger.info(f"Dataset already stored for {geojson.stem}, {year}. Skipping.")
                continue

            date_to_l1c_data = download_l1c_data(year, geojson)  # l1c data has shape (height, width, 13)
            date_to_l2a_data = download_l2a_data(year, geojson)  # l2a data has shape (height, width, 3)

            # Only keep dates where we have both L1C and L2A - they should be almost the same
            dates = list(set(date_to_l1c_data.keys()).intersection(date_to_l2a_data.keys()))
            dates.sort()
            assert abs(len(dates) - len(date_to_l1c_data)) <= 5  # Occasionally there is a small discrepancy
            assert abs(len(dates) - len(date_to_l2a_data)) <= 5

            # Now stack each to create 4D numpy array, with date dimension first
            l1c_data = np.stack([date_to_l1c_data[key] for key in dates], axis=0)
            l2a_data = np.stack([date_to_l2a_data[key] for key in dates], axis=0)

            ds = construct_xarray_dataset(
                geojson_path=geojson,
                dem_data=dem_data,
                l1c_data=l1c_data,
                l2a_data=l2a_data,
                dates=dates
            )

            ds.to_netcdf(OUTDIR / dataset_filename(geojson, year))  # Save as netCDF file

            if VISUALISE_DATA:
                plot_ds(ds, OUTDIR / "visualisations")

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

