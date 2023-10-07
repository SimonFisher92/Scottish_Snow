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


def patch_download_with_sentinelsat(patch_geojson_path: Path,
                                    username: str,
                                    password: str,
                                    download_dir: Path) -> None:

    """
    This function may be used to get snowpatch images, based on the 9 sentinel bands available, using the
    sentinelsat api for simplicity

    VERY IMPORTANT: this resolution of data is stored in the long term archive, these requests retrive from
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
                         date=('20230601', '20230630'),
                         platformname='Sentinel-2',
                         cloudcoverpercentage=(0, 50))

    # Print found products
    print(f"{len(products)} products found.")

    #print(products)

    # Download a specific product (or modify to download all products)
    # This will download the first product from the list
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

    # Read your GeoJSON into a GeoDataFrame
    aoi_gdf = gpd.read_file(patch_geojson_path)


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
        with rasterio.open('clipped_image.tif', "w", **out_meta) as dest:
            dest.write(out_image)

def validate_image(path):
    with rasterio.open(path, 'r') as src:
        image_data = src.read(1)
        # Plot the image data using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(image_data, cmap='gray')  # Change the colormap if needed
        plt.colorbar()
        plt.title('JP2 Satellite Image')
        plt.show()




if __name__ == "__main__":
    # patch_download_with_sentinelsat(
    #     patch_geojson_path=Path('An_Stuc.geojson'),
    #     username='YOURS',
    #     password= 'YOURS',
    #     download_dir= Path("../../data/snow_patches/An_Stuc")
    # )

    clip_sentinel_tile_to_region(
        tile_path=Path(r"C:\Users\Ultimate Gaming Comp\PycharmProjects\Scottish_Snow\data\snow_patches\An_Stuc\S2A_MSIL1C_20230626T113321_N0509_R080_T30VVH_20230626T151201.SAFE\GRANULE\L1C_T30VVH_A041833_20230626T113321\IMG_DATA\T30VVH_20230626T113321_B06.jp2"),
        patch_geojson_path=Path('An_Stuc.geojson')
    )

    validate_image(r"C:\Users\Ultimate Gaming Comp\PycharmProjects\Scottish_Snow\data\snow_patches\An_Stuc\S2A_MSIL1C_20230626T113321_N0509_R080_T30VVH_20230626T151201.SAFE\GRANULE\L1C_T30VVH_A041833_20230626T113321\IMG_DATA\T30VVH_20230626T113321_B06.jp2")
