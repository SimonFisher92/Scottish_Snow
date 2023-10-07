import logging
import argparse
from pathlib import Path

from src.measure_cls_band.regions import get_rois
from src.measure_cls_band.measurements import extract_time_series


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="A script to measure snow extent using the 20m CL band from Sentinel-2 imagery")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to downloaded jp2 data. This should be a directory full of .SAFE data.")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to output directory")

    args = parser.parse_args()

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    return args


def main():
    logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] [%(name)s] [%(levelname)s]: %(message)s")
    args = parse_args()

    logger.info("Starting SCL measurement code")

    rois = get_rois()

    logger.info(f"Measuring snow over {len(rois)} regions: {list(rois.keys())}")

    df = extract_time_series(rois, args)
    df.to_csv(Path(args.output_dir) / "cls_time_series.csv")

    logger.info("Code completed normally")


if __name__ == "__main__":
    main()
