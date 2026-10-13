from dataclasses import dataclass
from pathlib import Path
from typing import Literal

OutputSpace = Literal["raw", "xyz", "srgb"]

@dataclass(frozen=True)
class RenderConfig:
    backend: Literal["manual", "rawpy"]
    local_mode: Literal['mix', 'blend']
    space: OutputSpace
    raw_root: Path | None

    @classmethod
    def from_args(cls, args) -> "RenderConfig":
        if args.backend == "rawpy":
            if args.raw_input is None:
                raise ValueError("--backend rawpy requires --raw-input")
            if args.space not in (None, "srgb"):
                raise ValueError("--backend rawpy always outputs sRGB (omit --space)")
            if args.local_mode == "blend":
                raise ValueError("--backend rawpy only supports --local-mode mix")
            return cls(backend="rawpy", space="srgb", local_mode="mix", raw_root=args.raw_input)
        # noinspection PyTypeChecker
        return cls(backend="manual", local_mode=args.local_mode or "blend", space=args.space or "raw", raw_root=args.raw_input)
