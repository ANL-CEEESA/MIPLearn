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

clean-csv:
	rm -rvf */*/*.csv

clean-png:
	rm -rfv */*/*.png

.PHONY: clean
.SECONDARY:
