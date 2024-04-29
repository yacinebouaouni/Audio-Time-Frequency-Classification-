import os
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from audiodetect.data.utils_data import load_audio_file, compute_log_mel_spectrogram
from omegaconf import OmegaConf
from pathlib import Path


def process_audio(species, root_path, audio_file, dst_path, config):
    audio = load_audio_file(os.path.join(root_path, species, audio_file))[
        : config.DURATION * config.FS
    ]
    spec = compute_log_mel_spectrogram(audio, config)
    np.save(os.path.join(dst_path, species, audio_file.replace(".ogg", "")), spec)


def preprocess(root_path, dst_path, config):
    with mp.Pool(config.N_JOBS) as pool:
        for species in os.listdir(root_path):
            if not os.path.exists(os.path.join(dst_path, species)):
                os.makedirs(os.path.join(dst_path, species))
            audio_files = os.listdir(os.path.join(root_path, species))
            total_files = len(audio_files)
            with tqdm(total=total_files, desc=f"Processing {species}") as pbar:
                pool.starmap(
                    process_audio,
                    [
                        (species, root_path, audio_file, dst_path, config)
                        for audio_file in audio_files
                    ],
                )
                pbar.update(total_files)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process audio files.")
    parser.add_argument(
        "--root_path", type=str, required=True, help="Path to the root directory of the audio files."
    )
    parser.add_argument(
        "--dst_path",
        required=True,
        type=str,
        help="Path to the destination directory for processed files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    config_path = Path(__file__).parent / "../config/data.yaml"
    config = OmegaConf.load(str(config_path))
    args = parse_arguments()
    root_path = args.root_path
    dst_path = args.dst_path
    preprocess(root_path, dst_path, config)
