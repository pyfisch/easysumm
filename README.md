Easysumm
========

File format of dataset
----------------------

Each entry in a dataset consists of two documents: a text and the corresponding summary.
A dataset split (e.g. training data) is stored in a pair of two files.
The first file contains all texts separated by newlines and its name ends in `.source`.
The second file contains all summaries in the same format and order as the first file and its name ends in `.target`.

To train or test the model the path and filename prefix is specified with tree different command line arguments:

* `--train_prefix`
* `--valid_prefix`
* `--test_prefix`

> Note: Having individual parameters for each split is more flexible
> than specifying a single data-dir which contains all three splits
> as it simplifies testing on multiple different test sets.

