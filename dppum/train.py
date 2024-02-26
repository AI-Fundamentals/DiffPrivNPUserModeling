import tensorflow as tf
from tensorflow_privacy import DPKerasAdamOptimizer
import numpy as np
import json
import os
import lab as B
import neuralprocesses.tensorflow as nps

from dppum.privacy_oracle import get_sigma_from_privacy_loss_distribution as get_sigma
from dppum.util import calc_cat_confidence, flatten_first_two_dims




def train_model_dp_tf(
    model,
    dataset_train,
    dataset_train_metadata,
    loss_fn,
    num_epochs,
    dataset_test=None,
    epsilon=1,
    delta=None,
    clipping_bound=2,
    optimizer_name='Adam',
    learning_rate=5e-4,
    dp_enc = True,
    dp_dec = False,
    num_samples=5,
    warmup_epoch = False,
    shuffle = True,
    model_save_dir = None
):
    
    """
    Train a neural process model with differential privacy by streaming data
    from a tensorflow dataset. The model is modified in-place.

    Parameters
    ----------
    model : neuralprocesses.tensorflow.Model
        The model to be trained.
    dataset_train : tensorflow training dataset
        The dataset to be used for training.
    dataset_train_metadata : dict
        A dictionary with keys 'n_users', 'n_minibatches' representing the
        metadata of dataset_train.
    loss_fn : callable
        The loss function to be used for training. Must be from the file
        dppum.loss
    num_epochs : int
        The number of epochs for training.
    epsilon : float, optional
        The privacy budget for differential privacy, by default 1.
    delta : float, optional
        The delta parameter for differential privacy, by default None, which
        then defaults to 1/(num_users)^2
    clipping_bound : float, optional
        The clipping bound c for the gradients, by default 2.
    optimizer_name : str, optional
        The name of the optimizer to be used, must be from this list:
        ['Adam']. By default 'Adam'.
    learning_rate : float, optional
        The learning rate for the optimizer, by default 5e-4.
    dp_enc : bool, optional
        Whether to use differential privacy for the encoder, by default True.
    dp_dec : bool, optional
        Whether to use differential privacy for the decoder, by default False.
    num_samples : int, optional
        The number of model samples from the latent space, by default 5.
    warmup_epoch : bool, optional
        Whether to use a warmup epoch, by default False. If True there will be
        one epoch (0) where model performance is assessed but the model is not
        trained.
    shuffle : bool, optional
        Whether to shuffle dataset_train before each epoch, by default True.
    model_save_dir : str, optional
        The directory where the trained model should be saved, by default None.

    Returns
    -------
    history : dict
        A dictionary with the training metrics: loss, accuracy, confidence.

    Notes
    -----
    The model is modified in-place so it is not returned.
    """
        
    # Before training, make a dictionary with the training arguments in and save it
    if model_save_dir:
        # Check if the directory exists
        if not os.path.exists(model_save_dir):
            # If not, create the directory
            os.makedirs(model_save_dir)
        
        # Make a dictionary of training arguments    
        training_args = {}
        training_args['model_name'] = model._name
        training_args['dataset_train_metadata'] = dataset_train_metadata
        training_args['loss_fn_name'] = loss_fn.__name__
        training_args['num_epochs'] = num_epochs
        training_args['epsilon'] = epsilon
        training_args['delta'] = delta
        training_args['clipping_bound'] = clipping_bound
        training_args['optimizer_name'] = optimizer_name
        training_args['learning_rate'] = learning_rate
        training_args['dp_enc'] = dp_enc
        training_args['dp_dec'] = dp_dec
        training_args['num_samples'] = num_samples
        training_args['warmup_epoch'] = warmup_epoch
        training_args['shuffle'] = shuffle
        training_args['model_save_dir'] = model_save_dir

        # Write to JSON file
        with open(os.path.join(model_save_dir, "training_loop_args.json"), 'w') as json_file:
            json.dump(training_args, json_file,indent=4)
    

    # Calculate privacy sigma
    if not delta:
        # If delta is not specified, the default is 1/(num_users^2)
        delta = 1 / (dataset_train_metadata['n_users'])**2
    
    # The number of times the gradients will be updated during training
    num_repeats = dataset_train_metadata['n_minibatches'] * num_epochs
    # The fraction of the data that is used in each minibatch
    subsampling_rate = 1/dataset_train_metadata['n_minibatches']
    # Calculate sigma
    sigma = get_sigma(epsilon, delta, num_repeats, subsampling_rate)


    # Setup optimizer
    valid_optimizers = ['Adam']
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        optimizer_priv = DPKerasAdamOptimizer(l2_norm_clip=clipping_bound,
                                          noise_multiplier=sigma,
                                         num_microbatches=1,
                                         learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name. Expected one of: {valid_optimizers}")
        
    
    
    # Define training metrics: loss, accuracy, and confidence of mean model category
    # These metrics are for within epochs, and are reset after each epoch
    accuracy_per_epoch = tf.keras.metrics.Accuracy()
    loss_per_epoch = tf.keras.metrics.Mean(name='elbo_loss')
    mean_confidence_per_epoch = tf.keras.metrics.Mean(name='mean_confidence')
    
    # Define lists to store the metrics for all epochs
    accuracy_all_epochs = []
    loss_all_epochs = []
    mean_confidence_all_epochs = []    
    
    # Use warmup epoch (or not)
    warmup_epoch = False
    if warmup_epoch:
        first_epoch = 0
    else:
        first_epoch = 1        
        
    print("Starting training loop")
    # Run the training loop
    for epoch in range(first_epoch,num_epochs+1):
        print(f"""######## Start of epoch {epoch} ########""")
        
        # Shuffle the training dataset. Again note that it is pre-batched so one
        # "sample" from the dataset is actually a minibatch.
        # If you have used a subset of the whole dataset, this still works
        # even though the number of batches will then be less than n_minibatches.
        if shuffle:
            dataset_train = dataset_train.shuffle(buffer_size=dataset_train_metadata['n_minibatches'])
        
        # Reset epoch metrics
        accuracy_per_epoch.reset_states()
        loss_per_epoch.reset_states()
        mean_confidence_per_epoch.reset_states()


        # Iterate over the batches of the training dataset.
        for step, (xc, yc, xt, yt) in enumerate(dataset_train):
            # Data are in a padded batch with a batch size of 1
            if tf.shape(xc)[0] == 1:
                xc = flatten_first_two_dims(xc)
                yc = flatten_first_two_dims(yc)
                xt = flatten_first_two_dims(xt)
                yt = flatten_first_two_dims(yt)
            
            # Transpose the y data so they go into the model
            if(tf.rank(yc)==4):
                # This should only happen if the data are re-batched
                yc_t = B.transpose(yc,perm=[0,1,3,2])
                yt_t = B.transpose(yt,perm=[0,1,3,2])
            elif(tf.rank(yc)==3):
                yc_t = B.transpose(yc,perm=[0,2,1])
                yt_t = B.transpose(yt,perm=[0,2,1])
            
           
            with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
                # Compute the loss value for this minibatch.
                state = B.global_random_state(B.dtype(xc))
                vector_loss = -loss_fn(
                        state=state,
                        model=model,
                        contexts=[(xc,yc_t)],
                        subsume_context=True,
                        xt=xt,
                        yt=yt_t,
                        normalise=False,
                        dtype_lik=tf.float32,
                        num_samples=num_samples,
                        )
                scalar_loss = tf.reduce_mean(vector_loss)
                

            # On the 0th epoch, do not train the model just run metrics for
            # untrained model. On other epochs, calculate and apply the gradients
            if epoch> 0 :
                if dp_enc:
                    # DP encoder gradients        
                    optimizer_priv.minimize(vector_loss, model.encoder.trainable_variables, tape=encoder_tape)
                else:
                    # Standard (non-private) gradients
                    encoder_gradients = encoder_tape.gradient(scalar_loss, model.encoder.trainable_variables)
                    optimizer.apply_gradients(zip(encoder_gradients, model.encoder.trainable_variables))
                
                if dp_dec:
                    # DP decoder gradients        
                    optimizer_priv.minimize(vector_loss, model.decoder.trainable_variables, tape=decoder_tape)
                else:
                    #Standard decoder gradients
                    decoder_gradients = decoder_tape.gradient(scalar_loss, model.decoder.trainable_variables)
                    optimizer.apply_gradients(zip(decoder_gradients, model.decoder.trainable_variables))
           

            # Update loss metric
            loss_per_epoch(scalar_loss)            
            
            # Assess accuracy after updating model gradients
            
            import pdb
            pdb.set_trace()
            mean, _, _, _ = nps.predict(
                model,xc, yc_t, xt, num_samples=num_samples
                )
           
            mean_cat = B.argmax(mean,1)
            yt_cat = B.argmax(yt,2)
            accuracy_per_epoch.update_state(yt_cat, mean_cat)
            
            # Keep track of confidence in model predictions
            mean_confidence_per_epoch(calc_cat_confidence(mean, cat_axis=-2))



        ##### End of epoch calculations #####

        # Append to epoch metrics
        loss_all_epochs.append(loss_per_epoch.result())
        accuracy_all_epochs.append(accuracy_per_epoch.result())
        mean_confidence_all_epochs.append(mean_confidence_per_epoch.result())
        
        # Print epoch metrics
        print(f"Loss: {np.round(float(loss_all_epochs[-1]),5)}")
        print(f"Accuracy of mean predictions: {np.round(float(accuracy_all_epochs[-1]),3)}")
        print(f"Confidence of mean predictions: {np.round(float(mean_confidence_all_epochs[-1]),3)}")
        
        
        if dataset_test:
            print("Running though test dataset")
            # Loop through test dataset
            for xc,yc,xt,yt in dataset_test:
                yc_t = B.transpose(yc,perm=[0,2,1])
                yt_t = B.transpose(yt,perm=[0,2,1])
            
            mean, _, _, _ = nps.predict(
                model,xc, yc_t, xt, num_samples=num_samples
                )
            accuracy_all_epochs.append(accuracy_per_epoch.result())
        
        
        if model_save_dir:
            # Check if the directory exists
            if not os.path.exists(model_save_dir):
                # If not, create the directory
                os.makedirs(model_save_dir)
            
            # Save the model
            model_name = f"weights_epoch_{epoch}.tf"
            model.save_weights(os.path.join(model_save_dir, model_name))

    
    ##### End of training loop #####
    
    # Return the history data (training metrics)
    history = {
        'loss' : loss_all_epochs,
        'cat_accuracy' : accuracy_all_epochs,
        'cat_confidence' : mean_confidence_all_epochs
        }
    return history

