# dp-priv-python
Differentially Private Probabilistic User Modelling (in python)


## Installation instructions
1. Create and activate a python 3.11 environment
2. Install packages from pip:

    ```
    pip install -r requirements.txt
    ```

    or on Mac:

    ```
    pip install -r requirements_mac.txt
    ```

3. Install `neuralprocesses` from github:

    ```
    pip install git+https://github.com/wesselb/neuralprocesses.git
    ```

4. Install nightly version of tensorflow probability:

    ```
    pip install tfp-nightly
    ```

## Example usage
For training with 6400 users

```python -m experiments.ex2.experiment2_train --num_batches 1600
```

