# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright © 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.
# Written by Alinson S. Xavier <axavier@anl.gov>

CHALLENGES := \
    stab/ChallengeA \
    knapsack/ChallengeA

main: $(addsuffix /performance.png, $(CHALLENGES))

%/training_data.bin:
	python benchmark.py train $*

%/benchmark_baseline.csv: %/training_data.bin
	python benchmark.py test-baseline $*

%/benchmark_ml.csv: %/benchmark_baseline.csv
	python benchmark.py test-ml $*

%/performance.png: %/benchmark_ml.csv
	python benchmark.py charts $*

clean:
	rm -rvf stab knapsack

clean-ml:
	rm -rvf */*/benchmark_ml.csv

clean-charts:
	rm -rfv */*/performance.png

.PHONY: clean clean-ml clean-charts
.SECONDARY:
