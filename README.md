# dp-priv-python
Differentially Private Probabilistic User Modelling (in python)


## Installation instructions
1. Create and activate a python 3.9 environment
2. Install packages from pip. On a system with an Nvidia GPU:

    ```
    pip install -r requirements_nvidia.txt
    ```

    or on M1/M2 Mac:

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

5. Test the environment installation:

    ```
    pytest tests/test_nps_tf.py
    ```
    

## Example usage
### Note about needing the training data

For training with 6400 users (4 per batch)

```
python -m experiments.ex2.experiment2_train --num_batches 1600
```

## Workflow
1. Run training script. Load training data from data folder. Save model weights and metadata parameters in models folder. Save training metrics plot to figures folder.
2. Run test script. Load test data from model folder. Load models weights from model folder and save test performance data to model folder.
3. Load test performance data and plot. Save to figures folder.