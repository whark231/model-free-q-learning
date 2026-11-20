"""Transaction costs with drift (post-hoc bonus extension).

docs/methods/TC_DRIFT_DESIGN.md is the design. spec.py is the frozen shared
interface. run.py trains and scores the final runs
(python -m src.tc_drift.run all). The earlier no-drift design
(src/transaction_costs.py) is kept unchanged as a labelled negative result.
"""
