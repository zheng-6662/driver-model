# v58 modular split

- `paths.py`: project paths, protocol paths, data/style/result path discovery, run directory creation
- `utils.py`: JSON IO, environment parsing, stdout tee, steering-unit conversion, generic curve utilities
- `config.py`: runtime hyperparameters, model/loss constants, smoke-mode overrides, optimizer/scheduler helpers
- `data.py`: protocol split, IO, teacher-state construction, sample building
- `modeling.py`: dataset, model definition, output unpacking
- `losses.py`: training losses and reversal/phase-aware loss helpers
- `metrics.py`: denormalization, head/tail/peak/trend metrics, structured reversal metrics
- `evaluation.py`: test-time plotting and artifact export
- `train.py`: orchestration entrypoint `main()`
- `shared.py` and `losses_metrics.py`: compatibility re-export layers for older imports
