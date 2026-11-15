# Contributing

Thanks for looking. This is a small teaching repo, so I keep the bar simple.

1. Open a pull request against `main`. Keep it to one change.
2. Run `pytest -q` before you push. CI runs the same tests on Ubuntu with no model download.
3. If you change anything that affects a number in `results/RESULTS.md`, re-run `./run_all.sh` and commit the new `results/` with it. Do not edit result tables by hand.
4. No model weights, adapters or secrets in commits. `outputs/` is ignored for that reason.
5. Write docs in plain first person. Say what did not work as well as what did.
