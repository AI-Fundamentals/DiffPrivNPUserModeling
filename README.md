# dp-priv-python

Differentially Private Probabilistic User Modelling (in python). This is a python implementation of [this](https://github.com/hamalajaa/DifferentiablyPrivateProbabilisticUserModeling) Julia repo.

#### Contact details

This repo was developed by [Jonathan Taylor](mailto:jonathan.taylormanchester.ac.uk), Research IT, University of Manchester.

Questions about the scientific content should be directed to the paper's corresponding author, [Hari Harikumar](mailto:haripriya.harikumar@manchester.ac.uk).

#### Installation instructions

1. Clone and enter the repo:
   
   ```shell
   git clone https://github.com/AI-Fundamentals/dp-priv-python.git
   cd dp-priv-python
   ```
   
   NB if using CSF, please ignore the rest of the instructions and follow the workflow in the [jobscripts folder](jobscripts/README.MD).

2. Create and activate a python 3.11 environment

3. Install packages from pip:
   
   ```shell
   pip install -r requirements.txt
   ```

4. Install PyTorch:
   
   - On a system with Nvidia GPU:
   
   ```shell
   pip install torch
   ```
   
   - On Mac with Silicon GPU:
   
   ```shell
   pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```
   
   - On high-performance computing systems, refer to the relevant documentation.

5. Test the environment installation:
   
   ```shell
   pytest tests/test_nps.py
   ```

6. Run unit tests:
   
   ```shell
   python -m pytest
   ```

## Training and evaluation

#### Training data

This repo requires pre-computed data which are generated using [this Julia code](https://github.com/AI-Fundamentals/DifferentiableUserModels-DataGen).

#### Example training

To run the training for experiment 2:

```shell
python -m experiments.ex2.experiment2_train -settings settings/settings_ex2_train.json
```

Settings must be loaded from a valid `json` file. If no valid file is found (or the `-settings` argument isn't used), the default settings will be loaded instead. For full details of the settings file, see the docstrings in [dppum/settings.py](dppum/settings.py). Each function in this file returns a dictionary with the same keys as are required in the relevant settings file.

#### Re-training a pre-trained model

In the training settings json file, there is an item 'init_weights'. Set this to the path of a file containing weights from a previous model training run. These weights will then be used as the initial training weights.

#### Example evaluation

To run the epochs evaluation (i.e. model performance vs number of training epochs) for experiment2:

```shell
python -m experiments.ex2.experiment2_eval_epochs -settings settings/settings_ex2_eval_epochs.json
```

This will then save a figure in your `figs_dir` folder from the settings file, and also the evaluation metrics will also be saved in the `models_dir` folder.

To run the n_traj evaluation (i.e. model performance vs number of context trajectories), run the experiment2_eval_ntraj script. *NB this is not yet fully implemented.*

## Workflow

1. Generate training data using the [Julia code](https://github.com/AI-Fundamentals/DifferentiableUserModels-DataGen).
2. Run training script. Load training data from data folder. Save model weights and metadata parameters in models folder. Save training metrics plot to figures folder.
3. Run evaluation script(s). Load test data from model folder. Load models weights from model folder and save test performance data to model folder.
4. Load test performance data and plot. Save to figures folder.

For a detailed description of the workflow, see the [jobscripts folder](/jobscripts/README.md).