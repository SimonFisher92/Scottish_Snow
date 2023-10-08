from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import time
from sentinelsat.exceptions import LTATriggered
from datetime import date
import rasterio
import matplotlib.pyplot as plt
from pathlib import  Path
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import shutil


def patch_download_with_sentinelsat(patch_geojson_path: Path,
                                    username: str,
                                    password: str,
                                    download_dir: Path) -> None:

    """

    I was exploring alternative methods to data downloading, this is no better than Murrays method and should
    not be called preferentially

    This function may be used to get snowpatch images, based on the 9 sentinel bands available, using the
    sentinelsat api for simplicity

    VERY IMPORTANT: this resolution of data is stored in the long term archive, these requests retrieve from
    the LTA, the first retrieval may take hours. Once one of us has retrieved a patch, it should stay
    out of the LTA and be much faster for others

    :param patch_geojson_path: path to any geojson, generate these by running generate_prerequisites.py
    :param username: copernics username (https://scihub.copernicus.eu/dhus/#/home)
    :param password: copernicus password
    :param download_dir: path to store images
    :return: None, saves images to directory
    """

    # Connect to the API
    api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')

    # Load your area of interest from the GeoJSON
    footprint = geojson_to_wkt(read_geojson(patch_geojson_path))

    # Search for available products
    # You can change the date and product type based on your needs
    products = api.query(footprint,
                         date=('20230301', '20230630'),
                         platformname='Sentinel-2',
                         cloudcoverpercentage=(0, 30))

    # Print found products
    print(f"{len(products)} products found.")


    # for dev purposes
    uuid = next(iter(products))

    max_retries = 5
    retry_delay = 3600  # retry after 1 hour


    for i in range(max_retries):
        try:
            product_info = api.download(uuid, directory_path=download_dir)
            print(f"Downloaded {product_info['title']}.")
            break
        except LTATriggered:
            print(f"Product is still in LTA. Retry {i + 1}/{max_retries}. Waiting for {retry_delay} seconds.")
            time.sleep(retry_delay)


def clip_sentinel_tile_to_region(tile_path: Path,
                                 patch_geojson_path: Path) -> None:
    """
    Sentinel images come in tiles of 100mk^2, they must be clipped to snowpatches using the geojsons
    the team has generated.

    VERY IMPORTANT:
    Here's a breakdown of the Sentinel-2 MSI bands and their respective resolutions:

    10m Resolution Bands:

    Band 2 (Blue)
    Band 3 (Green)
    Band 4 (Red)
    Band 8 (NIR)
    20m Resolution Bands:

    Band 5 (Red Edge 1)
    Band 6 (Red Edge 2)
    Band 7 (Red Edge 3)
    Band 8A (Red Edge 4)
    Band 11 (SWIR 1)
    Band 12 (SWIR 2)
    60m Resolution Bands:

    Band 1 (Coastal/Aerosol)
    Band 9 (Water Vapor)
    Band 10 (Cirrus)

    Therefore, to load in the 10m res bands be sure to load ".....1_B02.jp2", ".....1_B03.jp2" etc

    :param tile_path: path to desired tile
    :param patch_geojson_path: path to geojson
    :return: None currently
    """

    # Read your GeoJSON into a GeoDataFrame
    aoi_gdf = gpd.read_file(patch_geojson_path)

    patch = aoi_gdf['name'][0]
    band = str(tile_path)[-7:-4]

    # Open the Sentinel image
    with rasterio.open(tile_path) as src:
        aoi_gdf = aoi_gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, aoi_gdf.geometry, crop=True)
        out_meta = src.meta.copy()

        # Update the metadata with new dimensions, transform, and CRS
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": aoi_gdf.crs.to_string()
        })

        # Save the clipped image
        with rasterio.open(f"{patch}_{band}_image.tif", "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"{patch} image at band {band} successfully clipped and written to local directory")

def validate_image(path: Path) -> None:

    """
    Simple plot function for output tiff
    :param path: path to generated tiff
    :return: None
    """

    filename = str(path).split('/')[-1]


    with rasterio.open(path, 'r') as src:
        image_data = src.read(1)
        # Plot the image data using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data, cmap='gray')  # Change the colormap if needed
        plt.colorbar()
        plt.title(f'{filename}')
        plt.show()




if __name__ == "__main__":
    # patch_download_with_sentinelsat(
    #     patch_geojson_path=Path('An_Stuc.geojson'),
    #     username='YOURS',
    #     password= 'YOURS',
    #     download_dir= Path("../../data/snow_patches/An_Stuc")
    # )

    clip_sentinel_tile_to_region(
        tile_path=Path(r"..\..\data/S2B_MSIL2A_20230415T114349_N0509_R123_T30VVJ_20230415T124132.SAFE/GRANULE/L2A_T30VVJ_A031895_20230415T114350/IMG_DATA/R10m/T30VVJ_20230415T114349_B02_10m.jp2"),
        patch_geojson_path=Path('Beinn_a_Bhuird.geojson')
    )

    validate_image(Path(r"Beinn_a_Bhuird_10m_image.tif"))
