# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2024-02-06

### Added

- Add ML strategies for user cuts
- Add ML strategies for lazy constraints

### Changed

- LearningSolver.solve no longer generates HDF5 files; use a collector instead.
- Add `_gurobipy` suffix to all `build_model` functions; implement some `_pyomo` and `_jump` functions.

## [0.3.0] - 2023-06-08

This is a complete rewrite of the original prototype package, with an entirely new API, focused on performance, scalability and flexibility.

### Added

- Add support for Python/Gurobipy and Julia/JuMP, in addition to the existing Python/Pyomo interface.
- Add six new random instance generators (bin packing, capacitated p-median, set cover, set packing, unit commitment, vertex cover), in addition to the three existing generators (multiknapsack, stable set, tsp).
- Collect some additional raw training data (e.g. basis status, reduced costs, etc)
- Add new primal solution ML strategies (memorizing, independent vars and joint vars)
- Add new primal solution actions (set warm start, fix variables, enforce proximity)
- Add runnable tutorials and user guides to the documentation.

### Changed

- To support large-scale problems and datasets, switch from an in-memory architecture to a file-based architecture, using HDF5 files.
- To accelerate development cycle, split training data collection from feature extraction.

### Removed

- Temporarily remove ML strategies for lazy constraints
- Remove benchmarks from documentation. These will be published in a separate paper.


## [0.1.0] - 2020-11-23

- Initial public release
