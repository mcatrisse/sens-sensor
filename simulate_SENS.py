import os
import sys
import zoneinfo
import time
import datetime
import argparse

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
import parameters as pm


from lib.functions_simulation import sensor_processing

tzinfo = zoneinfo.ZoneInfo(time.tzname[0])

parser = argparse.ArgumentParser(description="Run SENSEmulator")
parser.add_argument("--audio_file", type=str, required=True,   help="Path to the audio file")
parser.add_argument("--seconds_segment", type=int, default=3, help="Seconds per audio chunk to analyse")
parser.add_argument("--n_segments", type=int, default=10, help="Number of segments to integrate")
args = parser.parse_args()

sensor_processing(
    audio_file_path=os.path.join("input", args.audio_file),
    saving_folder_path="output",
    gain=1,
    timestamp=datetime.datetime.now(tzinfo).replace(microsecond=0),
    action="save",
    seconds_segment=args.seconds_segment,
    n_segments=args.n_segments,
    model_CLAP_path=pm.model_CLAP_path,
    pca_path=pm.pca_path,
    models_predictions_path=pm.models_predictions_path,
)