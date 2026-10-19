import argparse
import os
from collections.abc import Iterator
from pathlib import Path

import colour
import cv2 as cv
import numpy as np
import torch

import src.utils as utils
from src import get_project_dir
from src.config import RenderConfig
from src.datasets import Sample
from src.datasets import iter_gehler_shi, iter_lsmi
from src.interactive_session import run_interactive
from src.output import finalize, show_samples
from src.render_manual import render_manual
from src.render_rawpy import render_rawpy
from src.wbalance import WBAlgorithm, required_illums, estimate_illuminant

WB_ALGORITHMS = {
    'noop': WBAlgorithm.DISABLED,
    'white_patch': WBAlgorithm.WHITE_PATCH,
    'grey_world': WBAlgorithm.GREY_WORLD,
    'illuminant1': WBAlgorithm.ILLUMINANT1,
    'illuminant2': WBAlgorithm.ILLUMINANT2,
    'illuminant3': WBAlgorithm.ILLUMINANT3,
    'illu_map_mean': WBAlgorithm.ILLU_MAP_MEAN,
    'illu_map_wmean': WBAlgorithm.ILLU_MAP_W_MEAN,
    'local_illu': WBAlgorithm.LOCAL_ILLU
}

MAX_UINT16 = 65535
MAX_UINT8 = 255

D50 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
SONY_N_PLACES = 1317

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default="Gehler-Shi", help="The dataset path (in the data folder) to use as input data to enhance")
    parser.add_argument('--output', default=None, help="The output subfolder (in data/)")
    parser.add_argument('--wbalgorithm', required=True, choices=list(WB_ALGORITHMS), help="The White Balancing Algorithm to be used to enhance the dataset")
    parser.add_argument('--backend', required=True, choices=['manual', 'rawpy'], default='manual', help="Pipeline backend to convert to sRGB")
    parser.add_argument('--space', choices=['raw', 'xyz', 'srgb'], default=None, help="Output colour space (manual backend, default raw). The rawpy backend always outputs sRGB.")
    parser.add_argument('--local-mode', choices=['blend', 'mix'], default=None, help="How local illuminants are combined (manual default: blend; rawpy: mix)")
    parser.add_argument('--raw-input', default="sony_superraw", help="RAW files directory [needed by rawpy backend]")
    parser.add_argument('--limit', type=int, default=None, help="Process only the first N samples")
    parser.add_argument('--show', type=int, default=0, metavar='N', help="Show the first N outputs at the end")
    parser.add_argument('--dry-run', action='store_true', default=False, help="Don't write anything")
    parser.add_argument('--interactive', action='store_true', default=False, help="allow interactive changes to the estimated illuminant by end-user")
    parser.add_argument('--order', choices=['shuffled', 'sorted'], default='sorted', help="whether the dataset should be shuffled or sorted before processing it.")
    return parser

def get_device() -> str:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'  # type: ignore
    print(f"Accelerator: {device}")
    return device


# noinspection D
def run(samples: Iterator[Sample], algorithm: WBAlgorithm, cfg: RenderConfig, out_dir: Path, args):
    need = required_illums(algorithm)

    def collect():
        """Yield (sample, est, png) for every processable sample."""
        done = 0
        for sample in samples:
            if args.limit is not None and done >= args.limit:
                break
            if sample.meta.n_illums < need:
                print(f"skipping {sample.name}: n_illums {sample.meta.n_illums} < {need}")
                continue
            if cfg.backend == "rawpy" and (sample.raw_path is None or not sample.raw_path.exists()):
                print(f"skipping {sample.name}: raw_path does not exist ({sample.raw_path})")
                continue

            png = utils.load_linear_rgb(sample.png_path)
            est = estimate_illuminant(algorithm, png, sample.meta, sample.dir_path, sample.place_i)
            done += 1
            yield sample, est, png

    if args.interactive:
        run_interactive([(sample, est) for sample, est, _ in collect()])
        return

    shown = []
    for sample, est, png in collect():
        if cfg.backend == "rawpy":
            final = render_rawpy(sample.raw_path, est)
        else:
            final = finalize(render_manual(png, est, sample.meta, cfg), cfg)

        print(f"done: {sample.name}")

        if len(shown) < args.show:
            shown.append(final)
        if not args.dry_run:
            cv.imwrite(str(out_dir / sample.name), final)

    if shown:
        show_samples(shown, title="white balanced samples", gamma=None if cfg.space == 'srgb' else 0.4)

def main():

    parser = build_parser()
    args = parser.parse_args()

    try:
        cfg = RenderConfig.from_args(args)
    except ValueError as e:
        parser.error(str(e))

    get_device()
    algorithm = WB_ALGORITHMS[args.wbalgorithm]
    proj_dir = get_project_dir()
    datapath = proj_dir / 'data' / args.input
    gehler = args.input == "Gehler-Shi"

    if gehler and cfg.backend == "rawpy":
        parser.error("Gehler-shi has no raw files: the rawpy backend supports LSMI only")
    if gehler and required_illums(algorithm) > 1:
        parser.error(f"Gehler-Shi is a single-illuminant dataset and does not support {args.wbalgorithm} which needs multiple illuminants")
    if (algorithm == WBAlgorithm.LOCAL_ILLU or cfg.backend == 'manual') and args.interactive:
        parser.error("--interactive flag is not compatible with local estimate white balancing algorithms or manual backend.")


    samples = iter_gehler_shi(datapath, args.order) if gehler else iter_lsmi(datapath, cfg.raw_root, args.order)

    out_dir = Path(os.path.join(proj_dir, 'data', (args.output or args.wbalgorithm)))
    if not args.dry_run and not args.interactive:
        out_dir.mkdir(parents=True, exist_ok=True)

    run(samples, algorithm, cfg, out_dir, args)


def hello_cv():
    print("OpenCV: ", cv.__version__)
    img = np.zeros((120, 400, 3), dtype=np.uint8)
    cv.putText(img, "OpenCV OK", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
    cv.imshow("hello", img)
    cv.waitKey(0)

if __name__ == "__main__":
    main()