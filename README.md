Easysumm
========

Finetuning
----------

The `finetune.py` script is used to finetune the model based on a pre-trained BART.

```
python finetune.py \
    --model_name bart-large \
    --train_prefix cnn_tiny/train \
    --valid_prefix cnn_tiny/val \
    --checkpoint_dir output/checkpoints/ \
    --model_dir output/models/
```

Required parameters are `--train_prefix` and `--valid_prefix` for the input data
and `--checkpoint_dir` and `--model_dir` to write checkpoints and more importantly
the final trained model.

See the help text of the script for the supported hyperparameters.

Evaluating
----------

Summaries are generated with `evaluate.py`.

```
python evaluate.py cnn_tiny/test.txt output/predictions.txt path/to/model/dir/
```

It takes three reuired parameters: a filename to read the test articles from,
a filename to write the predictions to and a path (or alternative name of) a
pre-trained summarization model.

See the help text for additional parameters.

File format of dataset
----------------------

Each entry in a dataset consists of two documents: a text and the corresponding summary.
A dataset split (e.g. training data) is stored in a pair of two files.
The first file contains all texts separated by newlines and its name ends in `.source`.
The second file contains all summaries in the same format and order as the first file and its name ends in `.target`.

To train the model the path and filename prefix is specified with two different command line arguments:

* `--train_prefix`
* `--valid_prefix`

> Note: Having individual parameters for each split is more flexible
> than specifying a single data-dir which contains all three splits
> as it simplifies testing on multiple different test sets.

