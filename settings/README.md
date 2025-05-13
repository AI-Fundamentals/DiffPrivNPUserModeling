# Settings files

Each setting file is a json with keys for relevant parameters for model training and evaluation. The files are then loaded in as dictionaries when used in the python scripts. Suggested default settings are contained within the example files. The main things you may wish to edit are the `num_users`, `num_epochs`, `models_dir`, and `figs_dir`.

## Train Settings

**Purpose**: Training the model.

**Examples files**: [experiment 1](ex1/settings_ex1_train.json), [experiment 2](ex2/settings_ex2_train.json), [experiment 3](ex3/settings_ex3_train.json)

**Settings function**: A dictionary of example settings is also returned by the function `dppum.default_settings_ex2_train()`

**Keys**:

- **num_users**: `int`
  - Number of users to load from the training data HDF.
- **batch_size**: `int`
  - Number of users to put into each batch.
- **train_hdf**: `str`
  - Path to the HDF file to load the training data from.
- **val_hdf**: `str`
  - Path to the HDF file to load the validation data from. If this is left blank, model validation accuracy will not be calculated as the model trains.
- **models_dir**: `str`
  - The folder to save the trained models.
- **figs_dir**: `str`
  - The folder for output figures.
- **init_weights**: `str`
  - Path to a file containing weights to initialize the model. If `null`/`None`, weights will be initialized randomly. An example would be `models/ex2/weights_epoch_5.pt`.
- **num_samples**: `int`
  - Number of samples to take for model evaluation.
- **num_epochs**: `int`
  - Number of training epochs.
- **epsilon**: `float`
  - Epsilon parameter in differential privacy.
- **clipping_bound**: `float`
  - L2 clipping bound parameter in differential privacy.
- **clip_grads_per_user**: `str`
  - Method to clip gradients per user. ('loop'/'vectorize'/'false').
- **learning_rate**: `float`
  - Learning rate for model training.
- **warmup_epoch**: `bool`
  - Use a warmup epoch 0 to test the untrained model.
- **dp_enc**: `bool`
  - Use differential privacy for the encoder during training.
- **dp_dec**: `bool`
  - Use differential privacy for the decoder during training.
- **shuffle**: `bool`
  - Shuffle the training dataset at the start of each epoch.
- **optimizer**: `str`
  - Name of the optimizer to be used for training. Valid values are: ['Adam'].
- **padding_value**: `float`
  - Value to use for padding during batching. Should be a value that is not in the scope of the real data (e.g. use `-1` for one-hot encoded categorical data).
- **nonlinearity**: `str`
  - Nonlinearity (i.e., activation function) in the neural process model. Must be 'ReLU' or 'LeakyReLU'.
- **likelihood**: `str`
  - Likelihood in the neural process model. Must be one of 'het' or 'lowrank'.
- **dim_lv**: `int`
  - Dimensionality of the latent variable in the neural process model.
- **lv_likelihood**: `str`
  - Likelihood of the latent variable. Must be one of `het`, `dense`, or `spikes-beta`.

## Eval epochs settings

**Purpose**: Evaluating the model accuracy vs number of training epochs.

**Example files**: [experiment 1](ex1/settings_ex1_eval_epochs.json),  [experiment 2](ex2/settings_ex2_eval_epochs.json), [experiment 3](ex3/settings_ex3_eval_epochs.json)

**Settings function**: A dictionary of example settings is also returned by the function `dppum.default_settings_ex2_eval_epochs()`

**Keys**:

- **num_users**: int
  - Number of users to load from the training data hdf.
- **batch_size**: int
  - Number of users to put into each batch. Default is 4.
- **eval_hdf**: str
  - Path to the hdf file to load the evaluation data from. This should normally have n_traj of 1 - 8, the same as the training data.
- **models_dir**: str
  - The folder to load the trained models from.
- **figs_dir**: str
  - The folder for output figures.
- **num_samples**: int
  - Number of samples to take for model evaluation.
- **padding_value**: float
  - Value to use for padding during batching. Should be a value that is not in the scope of the real data (e.g. use `-1` for one-hot encoded categorical data).

## Eval ntraj settings

**Purpose**: Evaluating the accuracy of a trained model vs number of context trajectories provided at inference.

**Example files**: [experiment 1](ex1/settings_ex1_eval_ntraj.json), [experiment 2](ex2/settings_ex2_eval_ntraj.json), [experiment 3](ex3/settings_ex3_eval_ntraj.json)

**Settings function**: A dictionary of example settings is also returned by the function `dppum.default_settings_ex2_eval_ntraj()`

**Keys**:

- **num_users**: int
  - Number of users to load from the training data hdf.
- **batch_size**: int
  - Number of users to put into each batch. Default is 4.
- **eval_hdf**: str
  - Path to the hdf file to load the evaluation data from for assessing the impact of n_traj. This should have n_traj = 10.
- **models_dir**: str
  - The folder to load the trained models from.
- **figs_dir**: str
  - The folder for output figures.
- **num_samples**: int
  - Number of samples to take for model evaluation.
- **padding_value**: float
  - Value to use for padding during batching. Should be a value that is not in the scope of the real data (e.g. use `-1` for one-hot encoded categorical data).
- **init_weights**: str
  - Path to a file containing weights to initialize the model. If null/None, weights will be initialized randomly. An example would be `models/ex2/weights_epoch_5.pt`.
- **experiment**: int
  - Experiment number. For example, must be `1` for experiment 1 or `2` for experiment 2. This is used to work out how to crop the data to the right dimensions.
