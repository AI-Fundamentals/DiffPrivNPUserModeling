
def default_settings_ex2():
    """Returns a dictionary of settings for training a model for experiment 2.
    
    Returns
    -------
    args_dict : dict
        A dictionary containing all the arguments for the function.
        The keys and values are as follows:
        num_epochs : int
            The number of epochs for training.
        epsilon : float
            The privacy budget for differential privacy.
        delta : float
            The delta parameter for differential privacy, which. If not specified
            then defaults to 1/(num_users)^2.
        clipping_bound : float
            The clipping bound c for the gradients.
        optimizer_name : str
            The name of the optimizer to be used, must be from this list:
            ['Adam'].
        learning_rate : float
            The learning rate for the optimizer.
        dp_enc : bool
            Whether to use differential privacy for the encoder.
        dp_dec : bool
            Whether to use differential privacy for the decoder.
        num_samples : int
            The number of model samples from the latent space.
        warmup_epoch : bool
            Whether to use a warmup epoch. If True there will be one epoch (0)
            where model performance is assessed but the model is not trained.
        shuffle : bool
            Whether to shuffle dataset_train before each epoch.
        models_dir : str
            The directory where the trained model will be saved.
        padding_values : float
            Padding value which will be discarded during the loss/accuracy calculations.
        clip_grads_per_user : str
            If set to 'false', model gradients will be calculated once per batch like normal.
            If set to 'loop', gradients will be calculated (and clipped if appropriate)
            for each user in the batch individually by looping through them.
    """ 
    
    settings = {
    "num_users": 128,
    "batch_size": 4,
    "train_hdf": "data/ex2/experiment2_training_data.hdf",
    "test_hdf": "data/ex2/experiment2_test_data.hdf",
    "models_dir": "models/ex2/",
    "figs_dir": "figures/ex2/",
    "num_samples": 1,
    "num_epochs": 5,
    "epsilon": 1.0,
    "delta": None,
    "clipping_bound": 2.0,
    "clip_grads_per_user": "loop",
    "learning_rate": 0.0005,
    "warmup_epoch": False,
    "clip_user": "loop",
    "dp_enc" : True,
    "dp_dec" : False,
    "shuffle" : False,
    "optimizer" : "Adam",
    "padding_value" : -1.
    }
    return settings