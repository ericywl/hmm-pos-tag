# ML Project

Implement Hidden Markov Model for POS tag.

The training and testing data are given in `EN`, `CN`, `FR` and `SG`.

## Usage

Preferably, you have `pyenv` or some Python version manager so you can
switch to Python 3.6.5. If not, use `python3` below. The same applies for 
`hmm2.py` which uses a second-order Hidden Markov Model.

```console
$ python hmm.py
```

The above command will iterate through the four given folders and do the
following:

1. Train the HMM model using `train` file
2. Predict the tags for `dev.in` file
3. Output the above prediction to `dev.test.out` file
4. Evaluate the prediction using the gold standard file `dev.out`

Note: `hmm.py` has two prediction modes. The default is using Viterbi. It can 
also be changed to use naive prediction. To do that, head to `hmm.py:424` and
change "viterbi" to "naive".
