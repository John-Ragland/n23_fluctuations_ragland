"""False-rejection rate of the outlier tests of tools.estimate_hydrophone_pattern on clean synthetic wavefronts
(18 arrivals, known pattern, no outliers). The arrival test targets a 1% family-wise rate."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
from test_tools import _pattern_wavefronts
from n23_fluctuations_ragland import tools
n = 100
arr = {"phase": 0, "gain": 0}; recs = {"phase": 0, "gain": 0}
for seed in range(n):
    p = tools.estimate_hydrophone_pattern(_pattern_wavefronts(np.random.default_rng(1000 + seed), n_arrivals=18))
    for k in arr:
        arr[k] += bool(p["rejected_arrivals"][k])
        recs[k] += sum(len(v) for v in p["rejected_records"][k].values())
print("runs with any rejected arrival:", arr, "of", n)
print("mean rejected records per run (of 414):", {k: v / n for k, v in recs.items()})
