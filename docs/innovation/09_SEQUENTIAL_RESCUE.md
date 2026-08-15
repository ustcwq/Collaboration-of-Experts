# Sequential Rescue Acquisition gate record

**Status: SKIPPED.** The sequential extension requires a frozen full-pool
selector and is meant to recover its accuracy at lower call cost. Neither DCRG
nor CPI passed source validation, so there is no valid full-pool innovation target
to approximate. The unit-cost convention does not remove this dependency.

No target labels were read, no stopping threshold was selected, and no accuracy-
call Pareto curve was generated. The offline simulator should be implemented
only after a selector clears both normal-pool and pool-shift source gates.

