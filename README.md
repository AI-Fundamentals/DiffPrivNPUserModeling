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
    

## Training and evaluation
#### Training data
This repo requires pre-computed data which are generated using [this Julia code](https://github.com/AI-Fundamentals/DifferentiableUserModels-DataGen).

#### Example training
To run the training for experiment 2:

```
python -m experiments.ex2.experiment2_train -settings settings/settings_ex2_train.json
```
Settings must be loaded from a valid `json` file. If no valid file is found (or the `-settings` argument isn't used), the default settings will be loaded instead. For full details of the settings file, see the docstrings in [dppum/settings.py](dppum/settings.py).

#### Re-training a pre-trained model
In the training settings json file, there is an item 'init_weights'. Set this to the path of a file containing weights from a previous model training run. These weights will then be used as the initial training weights.

#### Example evaluation
To run the epochs evaluation (i.e. model performance vs number of training epochs) for experiment2:
```
python -m experiments.ex2.experiment2_eval_epochs -settings settings/settings_ex2_eval_epochs.json
```

This will then save a figure in your `figs_dir` folder from the settings file, and also the evaluation metrics will also be saved in the `models_dir` folder.

To run the n_traj evaluation (i.e. model performance vs number of context trajectories), run the experiment2_eval_ntraj script. *NB this is not yet fully implemented.*

## Workflow
1. Run training script. Load training data from data folder. Save model weights and metadata parameters in models folder. Save training metrics plot to figures folder.
2. Run evaluation script(s). Load test data from model folder. Load models weights from model folder and save test performance data to model folder.
3. Load test performance data and plot. Save to figures folder.