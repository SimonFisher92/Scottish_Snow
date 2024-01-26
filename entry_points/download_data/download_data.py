from src.download.main import main
from src.produce_geojsons.produce_geojsons import write_geojson_bbox
import yaml
import json

with open("download_data_configuration.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# open secret copernicus keys json
with open('secret_copernicus_keys.json') as f:
    keys = json.load(f)

with open('my_coordinates.json') as f:
    coords = json.load(f)

write_geojson_bbox(coords)

#append keys to config
config['client_id'] = keys['client_id']
config['client_secret'] = keys['client_secret']

main(config=config)