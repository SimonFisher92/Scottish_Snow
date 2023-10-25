# Scottish_Snow

The purpose of Scottish_Snow is to generate a process by which we can characterise the intra and inter annual changes in snow patch area within Scotland, using satellite imagery. Given enough longitudional data, this can have implications for understanding climate change within Scotland, but it also serves as a useful way of estimating where and how patches are melting year-to-year.

We can break the task down into smaller issues like this:

## 1) Data harvesting

Where do we get the data from? Sentinal seems to be a great choice. We beleive we can batch download data from sentinel via their API. We need to explore the use of Sentinel. Currently, both Murray and Eddie are looking into this.

## 2) Key Coordinates

Where do we look? Gaining a method of data harvesting, we also need to agree on useful coordinates, and useful resolution (zoom). I know Eddie is an expert on this, Iain Cameron also may be someone to speak with. Its my feeling that as many coordinates as possible, spread over as many different regions of the highlands is the best approach. When I created Bluebird in 2020 (a weak version of what we are attempting to do here), Iain gave me a list of coordinates, I will add them to this repo as it might be a useful starting point.

Old bluebird data:
https://www.dropbox.com/sh/ti2esug7wzvqq2j/AAD_OuGe170Qb6rqGtpdTwwwa?dl=0

## 3) Segmentation

Having experience here, I can tell you that the primary nemisis of this kind of analysis are clouds. Ostensibly, they look like snow patches. How do we segment snow out from cloud? I had a previous approach which leveraged the subtle differences in reflective wavelength at certain sentinel bandwidths, but i suspect there is a better way. 

Eddie is looking at this at the moment, we are working at the lower end of "usefulness", as patches can become very small. Murray has found that sentinel provide segmentation masks for snow/ice with their data. Should we reinvent the wheel or use this? Its possible that the automatically generated segmentation masks are not of high enough resolution to catch the smallest patches.

## 4) Analysis

Scotland is cloudy. On average, per year, the satellites are over cloud-free patches around 10 times. This isnt a great amount. Previously, I fitted a trapezoid area-under to calculate the yearly area. Is there a better way? Most likely. Nevis Gullys are steep, Ciste Mhearad is not. Inevitably therefore, area data will be incorrectly weighted to the gently sloping patches around the cairngorms, even though cross-sectional area at perpendicular wouldnt reflect that.

## 5) Communication

How do we convey our results? I really like Streamlit. Its a lightweight webapp deployer in python, which you can host for free using the streamlit website. I imagine a webapp where people can select a patch, and look at the yearly, and seasonal changes. There should also be a page which summarises all of the patches. I think an app is key. Previously, I had a dropbox for data, but interactive is almost always better.

# This Repo

1) Please create an issue for what you intend to do (I have added a few starter issues)
2) Please do your work on a branch
3) Do not merge your own work onto master
4) If you submit a pull request, please provide details
5) Please be aware that it may take time for this repo to progress due to the commitments of repo members
6) Anyone and everyone is welcome to join and contribute to this repo, but we should be aware of protecting the master branch



# User guide
---

## Environment setup

See requirements.txt. This can be installed in a virtual environment with `pip install -r requirements.txt`


## Data download

Running python -m src.download.main -h displays:

	A script to download Sentinel-2 data for snowpatch analysis. Example command: python -m src.download.main --data_dir='data' --geojson_path='input/cairngorms_footprint.geojson'
	--product_filter='*B0[234]_10m.jp2' --target_tile='T30VVJ' --api_user=<> --api_password=<>
	
	optional arguments:
	  -h, --help            show this help message and exit
	  --data_dir DATA_DIR   Path to dir where data should be cached
	  --geojson_path GEOJSON_PATH
	                        Path to geojson file containing polygons covering all areas which data should be downloaded for
	  --num_threads NUM_THREADS
	                        Number of concurrent download threads. Default is 0 (no concurrency).
	  --product_filter PRODUCT_FILTER
	                        Path filter which is passed to sentinelsat.SentinelAPI.download(). The default is no filter. For all data use '*'. For 10m resolution RGB use '*B0[234]_10m.jp2'. See
	                        documentation at https://sentinelsat.readthedocs.io/en/latest/api_overview.html#downloading-parts-of-products.
	  --target_tile TARGET_TILE
	                        Optional field to restrict data to a single tile, such as T30VVJ
	  --max_cloud_cover MAX_CLOUD_COVER
	                        Only get results with total cloud cover % less than this
	  --month_range MONTH_RANGE
	                        Only get results from months in this range. Should be a string such as 4-10
	  --year YEAR           Only get results from this year. Should be a string such as 2020
	  --api_user API_USER   Username for Copernicus Sentinel API
	  --api_password API_PASSWORD
	                        Password for Copernicus Sentinel API
	

We recommend using `--num_threads=0` for now, as the concurrency may exhibit an intermittent bug.


### Example commands:

Get all cairngorms data from 2023, when cloud cover was below 50%:

	python -m src.download.main --data_dir='/media/murray/BE10-C259/data/Scottish_Snow' --geojson_path='input/cairngorms_footprint.geojson' --product_filter='' --num_threads=0 --target_tile='T30VVJ' --api_user="" --api_password="" --max_cloud_cover=50 --year=2023

Get all the cairngorms 20m SCL band:

	python -m src.download.main --data_dir='data' --geojson_path='input/cairngorms_footprint.geojson' --product_filter='*SCL_20m.jp2' --target_tile='T30VVJ' --api_user="" --api_password=""

Get all the cairngorms 10m RGB bands:

	python -m src.download.main --data_dir='data' --geojson_path='input/cairngorms_footprint.geojson' --product_filter='*B0[234]_10m.jp2' --target_tile='T30VVJ' --api_user="" --api_password=""



## Analysis

After downloading, you can run

	python -m src.measure_cls_band.main --data_dir="data"

This will create an "output" dir, and place summary plots and a csv here, which aggregates the snow area vs. time for each ROI.

Finally, see notebooks/scl\_time\_series\_analysis.ipynb for an example of plotting these time series, which looks a bit like this:

![timeseries_beinn_dearg](https://github.com/SimonFisher92/Scottish_Snow/assets/11088372/2c402240-73a5-401a-8968-8ec72298f8f3)

