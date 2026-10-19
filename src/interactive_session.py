from collections import OrderedDict

from src.datasets import Sample
from src.interactive_output import interactive
from src.render_rawpy import RawpyRenderer
from src.wbalance import IlluminantEstimate, ill2gains


def run_interactive(entries: list[tuple[Sample, IlluminantEstimate]]) -> None:
    """
    Cycle through (sample, est) entries in the trackbar UI.

    Renderers are cached in a Less Recently Used cache of 2 slots.
    """
    if not entries:
        print("no samples available for interactive output")
        return

    cache: OrderedDict[str, RawpyRenderer] = OrderedDict()

    def render_fn(sample):
        def frame(gains):
            render = cache.get(sample.name)
            if render is None:
                if len(cache) >= 2:
                    _k, old_render = cache.popitem(last=False)  # last=False => FIFO order pop
                    old_render.close()
                render = RawpyRenderer(sample.raw_path)
                cache[sample.name] = render
            return render.render(1.0 / gains)
        return frame

    try:
        interactive([(sample.name, ill2gains(est.mixed()), render_fn(sample)) for sample, est in entries])
    finally:
        for r in list(cache.values()):
            r.close()