
def default_settings_ex2():
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
    "clipping_bound": 2.0,
    "learning_rate": 0.0005,
    "warmup_epoch": False,
    "clip_user": "loop",
    "dp_enc" : True,
    "dp_dec" : False,
    "shuffle" : False,
    "optimizer" : "Adam",
    "padding_values" : -1.
    }
    return settings