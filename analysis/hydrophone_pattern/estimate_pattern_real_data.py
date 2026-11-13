"""Estimate the hydrophone pattern from the real tilt-removed wavefront values in memory, print the report and
save hydrophone_pattern.png here (the CLI step writes the same to dataproducts/)."""
import pickle
import time
from pathlib import Path

from n23_fluctuations_ragland import tools, cli
with tools.wavefront_values_path(True).open("rb") as f:
    wv = pickle.load(f)
t0 = time.time()
p = tools.estimate_hydrophone_pattern(wv)
print(f"{time.time()-t0:.1f} s")
print(cli.hydrophone_pattern_report(p))
fig = cli.plot_hydrophone_pattern(p)
fig.savefig(Path(__file__).parent / "hydrophone_pattern.png", dpi=150, bbox_inches="tight")
