
def default_settings_ex2_train():
    """Returns a dictionary of settings for training a model for experiment 2.
    
    Returns
    -------
    settings : dict
        A dictionary containing all the arguments for the function.
        The keys and values are as follows:
        num_users : int
            Number of users to load from the training data hdf.
        batch_size : int
            Number of users to put into each batch. Default is 4.
        train_hdf : str
            Path to the hdf file to load the training from.
        val_hdf : str
            Path to the hdf file to load the val from.
        models_dir : str
            The folder to save the trained models.
        figs_dir : str
            The folder for output figures.
        num_samples : int
            Number of samples to take for model evaluation.
        num_epochs : int
            Number of training epochs.
        epsilon : float
            Epsilon parameter in differential privacy.
        clipping_bound : float
            L2 clipping bound parameter in differential privacy.
        clip_grads_per_user : str
            Method to clip gradients per user. ('loop'/'vectorize'/'false').
        learning_rate : float
            Learning rate for model training.
        warmup_epoch : bool
            Use a warmup epoch 0 to test the untrained model.
        dp_enc : bool
            Use differential privacy for the encoder during training.
        dp_dec : bool
            Use differential privacy for the decoder during training.
        shuffle : bool
            Shuffle the training dataset at the start of each epoch.
        optimizer : str
            Name of the optimizer to be used for training. Valid values are:
            ['Adam'].
        padding_value : float
            Value to use for padding during batching. Should be a value that is
            not in the scope of the real data (e.g. use -1 for one-hot encoded
            categorical data.)
        "nonlinearity" : str
            Nonlinearity (i.e. activation function) in the neural proces model.
            Must be 'ReLU' or 'LeakyReLU'.
        "likelihood" : str
            Likelihood in the neuralprocess model. Must be one of “het” or
            “lowrank”.    
        dim_lv : int
            Dimensionality of the latent variable in the neural process model.
        "lv_likelihood" : str
            Likelihood of the latent variable. Must be one of “het”, “dense”,
            or “spikes-beta”.
    """ 
    
    settings = {
    "num_users": 128,
    "batch_size": 4,
    "train_hdf": "data/ex2/experiment2_training_data.hdf",
    "val_hdf": "data/ex2/experiment2_val_data.hdf",
    "models_dir": "models/ex2/",
    "figs_dir": "figures/ex2/",
    "num_samples": 1,
    "num_epochs": 5,
    "epsilon": 1.0,
    "delta": None,
    "clipping_bound": 2.0,
    "clip_grads_per_user": "loop",
    "learning_rate": 0.0005,
    "warmup_epoch": True,
    "dp_enc" : True,
    "dp_dec" : False,
    "shuffle" : False,
    "optimizer" : "Adam",
    "padding_value" : -1.0,
    "nonlinearity" : "LeakyReLU",
    "likelihood" : "het",
    "dim_lv" : 0,
    "lv_likelihood" : "het",
    }
    return settings



def default_settings_ex2_test():
    """Returns a dictionary of settings for testing a model for experiment 2.
    
    Returns
    -------
    settings : dict
        A dictionary containing all the arguments for the function.
        The keys and values are as follows:
        num_users : int
            Number of users to load from the training data hdf.
        batch_size : int
            Number of users to put into each batch. Default is 4.
        test_hdf : str
            Path to the hdf file to load the test from.
        models_dir : str
            The folder to load the trained models from.
        figs_dir : str
            The folder for output figures.
        num_samples : int
            Number of samples to take for model evaluation.
        padding_value : float
            Value to use for padding during batching. Should be a value that is
            not in the scope of the real data (e.g. use -1 for one-hot encoded
            categorical data.)
        
    """ 
    
    settings = {
        "num_users": 128,
        "batch_size": 4,
        "test_hdf": "data/ex2/experiment2_test_data.hdf",
        "models_dir": "models/ex2/",
        "figs_dir": "figures/ex2/",
        "num_samples": 1,
        "padding_value" : -1.0
    }
    return settings




