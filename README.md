# dp-priv-python
Differentially Private Probabilistic User Modelling (in python)


#### Installation instructions
1. Create and activate a python 3.9 environment
2. Install packages from pip:

    ```
    pip install -r requirements.txt
    ```
2. Install PyTorch:
   1. On a "standard" system with Nvidia GPU:
   ```
    pip install torch
    ```
   2. On Mac with Silicon GPU:

    ```
    pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
    ```
   3. On high-performance computing systems, refer to the relevant documentation.

4. Install `neuralprocesses` from github:

    ```
    pip install git+https://github.com/wesselb/neuralprocesses.git
    ```

6. Test the environment installation:

    ```
    pytest tests/test_nps.py
    ```
    


#### Training data
This repo requires pre-computed data which are generated using [this Julia code](https://github.com/AI-Fundamentals/DifferentiableUserModels-DataGen).

#### Example usage
To run the training for experiment 2:

```
python -m experiments.ex2.experiment2_train -settings settings/settings_ex2_train.json
```
Settings must be loaded from a valid `json` file. If no valid file is found (or the `-settings` argument isn't used), the default settings will be loaded instead. For full details of the settings file, see the docstrings in [dppum/settings.py](dppum/settings.py).


## Workflow
1. Run training script. Load training data from data folder. Save model weights and metadata parameters in models folder. Save training metrics plot to figures folder.
2. Run test script. Load test data from model folder. Load models weights from model folder and save test performance data to model folder.
3. Load test performance data and plot. Save to figures folder.