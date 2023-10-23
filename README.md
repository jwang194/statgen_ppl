# statgen_ppl

This is a repository containing some simple scripts which apply Tensorflow's sharding framework to statistical genetics models. 

The sharding is done in _runner.py_, while the models themselves are specified in _models.py_.

To perform benchmarking, execute the master scripts as follows:

_./[model]\_benchmark.sh [n\_GPU] [min_N] [max_N] [min_M] [max_M]_

where the first parameter describes the number of GPUs available to the pipeline, and the rest govern the range of exponents for the data sizes (samples and features, respectively). All results are stored in .hdf5 files in the data directory.

Currently, the standard linear mixed model (_lmm_) and Border et al's assortative mating model (_am_) are available. More models may be implemented in the future. 
