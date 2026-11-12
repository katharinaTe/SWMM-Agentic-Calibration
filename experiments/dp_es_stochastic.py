"""Detention pond, event score, stochastic setting (paper: Figure 8).

Corresponds to the notebook `dp_es_stochastic_gpt.ipynb` (v1.0.0).
Run from the folder that contains the input files, e.g. the repository root:

    python -m experiments.dp_es_stochastic
"""

import time
import resource

time_start = time.perf_counter()

# importing the module seeds the random number generators, as in the notebook
from swmm_agentic import detention_pond_event_score as experiment  # noqa: E402

experiment.EXPERIMENT_MODE = "stochastic"
experiment.run_experiment()

time_elapsed = (time.perf_counter() - time_start)
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ("%5.1f secs %5.1f MByte" % (time_elapsed,memMb))
