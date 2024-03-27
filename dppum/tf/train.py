import tensorflow as tf
import numpy as np
import json
import os
import lab as B
import neuralprocesses.tensorflow as nps

from dppum.privacy_oracle import get_sigma_from_privacy_loss_distribution as get_sigma
from dppum.util import calc_cat_confidence, flatten_first_two_dims, calc_cat_acc_onehot



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
    model_save_dir = None,
    padding_values = None,
    clip_grads_per_user = False
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
    padding_values : float, optional
        Padding value which will be discarded during the loss/accuracy calculations.
    clip_grads_per_user : str, optional
        If set to 'false', model gradients will be calculated once per batch like normal.
        If set to 'loop', gradients will be calculated (and clipped if appropriate)
        for each user in the batch individually by looping through them.
        If set to 'vectorize', gradients will be calculated for each user using a
        tf.vectorized_map, which is normally faster than loop for large batch
        sizes but slower for small batch sizes. Default is 'false'.

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
    num_repeats = dataset_train_metadata['n_batches'] * num_epochs
    # The fraction of the data that is used in each minibatch
    subsampling_rate = 1/dataset_train_metadata['n_batches']
    # Calculate sigma
    sigma = get_sigma(epsilon, delta, num_repeats, subsampling_rate)


    # Setup optimizer
    valid_optimizers = ['Adam']
    if optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name. Expected one of: {valid_optimizers}")
        
    
    
    # Define training metrics: loss, accuracy, and confidence of mean model category
    # These metrics are for within epochs, and are reset after each epoch
    train_accuracy_per_epoch = tf.keras.metrics.Mean(name='train_accuracy')
    loss_per_epoch = tf.keras.metrics.Mean(name='elbo_loss')
    mean_confidence_per_epoch = tf.keras.metrics.Mean(name='mean_confidence')
    if dataset_test:
        test_accuracy_per_epoch = tf.keras.metrics.Mean(name='test_accuracy')
    
    # Define lists to store the metrics for all epochs
    train_accuracy_all_epochs = []
    loss_all_epochs = []
    mean_confidence_all_epochs = []    
    if dataset_test:
        test_accuracy_all_epochs = []
    
    # Use warmup epoch (or not)
    if warmup_epoch:
        first_epoch = 0
    else:
        first_epoch = 1        
        
        
    def loss_wrapper(args_tuple):
        # This is a wrapper function to calculate the loss
        # It is written in a way so it can be used in tf.vectorized_map to 
        # calculate and clip the gradients on a per-user basis efficiently
        
        # Unwrap the input data into a batch size of 1
        xc,yc,xt,yt = args_tuple
        xc = tf.expand_dims(xc, 0)
        yc = tf.expand_dims(yc, 0)
        xt = tf.expand_dims(xt, 0)
        yt = tf.expand_dims(yt, 0)
        
        # We use one gradient tape for the encoder and one for the decoder to separate out the gradients
        with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
            # Compute the loss value for this minibatch.
            state = B.global_random_state(B.dtype(xc))
        
            vector_loss = -loss_fn(
                    state=state,
                    model=model,
                    contexts=[(xc,yc)],
                    subsume_context=True,
                    xt=xt,
                    yt=yt,
                    normalise=False,
                    dtype_lik=tf.float32,
                    num_samples=num_samples,
                    padding_values=padding_values
                    )
            scalar_loss = B.mean(vector_loss)
        
        # Update loss metric
        loss_per_epoch(scalar_loss)
        
        encoder_gradients = encoder_tape.gradient(scalar_loss, model.encoder.trainable_variables)           
        decoder_gradients = decoder_tape.gradient(scalar_loss, model.decoder.trainable_variables)
        
        # Perform L2 clipping
        if dp_enc:                               
            # Clip encoder gradients per user
            for i, grad in enumerate(encoder_gradients):
                # L2 clipping
                grad_normalized = tf.norm(grad)
                encoder_gradients[i] = grad * tf.minimum(1.0,(clipping_bound/grad_normalized))
        if dp_dec:                               
            # Clip encoder gradients per user
            for i, grad in enumerate(decoder_gradients):
                # L2 clipping
                grad_normalized = tf.norm(grad)
                decoder_gradients[i] = grad * tf.minimum(1.0,(clipping_bound/grad_normalized))
        
        return encoder_gradients, decoder_gradients
    
    def loss_wrapper_wrapper(loss_wrapper,args):
        return tf.vectorized_map(loss_wrapper, args)
        
    print("Starting training loop")
    # Run the training loop
    for epoch in range(first_epoch,num_epochs+1):
        print(f"""######## Start of epoch {epoch} ########""")
        
        # Shuffle the training dataset
        if shuffle:
            dataset_train = dataset_train.shuffle(buffer_size=dataset_train_metadata['n_batches'])
        
        
        # Reset epoch metrics
        train_accuracy_per_epoch.reset_states()
        loss_per_epoch.reset_states()
        mean_confidence_per_epoch.reset_states()
        if dataset_test:
            test_accuracy_per_epoch.reset_states()



        # Iterate over the batches of the training dataset.
        for step, (xc, yc, xt, yt) in enumerate(dataset_train):
            
            # First, calculate the loss/gradients
            # (the loss metric is updated within loss_wrapper so we don't need to see it explicitly here)
            
            if clip_grads_per_user == 'vectorize':
                # Use tf.vectorized_map to calculate and clip (if appropriate) gradients on a per-user basis
                encoder_gradients_batch, decoder_gradients_batch = tf.vectorized_map(
                    loss_wrapper, (xc, yc, xt, yt))
                
                encoder_gradients = [B.mean(B.stack(*gradients_list), axis=0) for gradients_list in encoder_gradients_batch]
                decoder_gradients = [B.mean(B.stack(*gradients_list), axis=0) for gradients_list in decoder_gradients_batch]
                
            elif clip_grads_per_user == 'loop':
                # Loop through users to calculate loss and clip (if appropriate) gradients on a per-user basis
                batch_size = B.shape(yt)[0]
                encoder_gradients_batch = []
                decoder_gradients_batch = []
                for i in range(batch_size):
                    xc_user = xc[i]
                    xt_user = xt[i]
                    yc_user = yc[i]
                    yt_user = yt[i]
                    gradients_batch = loss_wrapper((xc_user, yc_user, xt_user, yt_user))
                    encoder_gradients_batch.append(gradients_batch[0])
                    decoder_gradients_batch.append(gradients_batch[1])

                # Average the gradients             
                encoder_gradients = []
                decoder_gradients = []
                for i in range(len(encoder_gradients_batch[0])):
                    tensors = [lst[i] for lst in encoder_gradients_batch]
                    mean_tensor = tf.reduce_mean(tensors, axis=0)
                    encoder_gradients.append(mean_tensor)
                for i in range(len(decoder_gradients_batch[0])):
                    tensors = [lst[i] for lst in decoder_gradients_batch]
                    mean_tensor = tf.reduce_mean(tensors, axis=0)
                    decoder_gradients.append(mean_tensor)
                
            else:
                # Calculate and clip (if appropriate) gradients on a per-batch basis
                encoder_gradients, decoder_gradients = loss_wrapper((xc, yc, xt, yt))
            
            if dp_enc or dp_dec:
                print("Need to add noise to gradients here")
            # We have now calculated the loss/gradients
            
            # Apply gradients to update model weights
            optimizer.apply_gradients(zip(encoder_gradients, model.encoder.trainable_variables))    
            optimizer.apply_gradients(zip(decoder_gradients, model.decoder.trainable_variables))
            
            # Assess accuracy after updating model gradients
            yt_pred, _, _, _ = nps.predict(
                model,xc, yc, xt, num_samples=num_samples
                )
            
            # Create mask for the padding
            if padding_values:
                # A mask for where the padding is. This is the same shape as the batch
                padding_mask = (yt == padding_values)
                # This is collapsed along the categorical dimension
                padding_mask = B.any(padding_mask,axis=-2)
            else:
                padding_mask = None
            
            # Accuracy of the non-padding values
            accuracy=calc_cat_acc_onehot(yt,yt_pred,cat_axis=-2,padding_values=padding_mask)
            # Mean confidence of the non-padding values
            confidence = calc_cat_confidence(yt_pred,-2,padding_mask)
            
            # Update accuracy and confidence metrics
            train_accuracy_per_epoch.update_state(accuracy)
            mean_confidence_per_epoch(confidence)

        ##### End of epoch calculations #####

        # Append to epoch metrics
        loss_all_epochs.append(loss_per_epoch.result())
        train_accuracy_all_epochs.append(train_accuracy_per_epoch.result())
        mean_confidence_all_epochs.append(mean_confidence_per_epoch.result())
        
        # Print epoch metrics
        print(f"Loss: {np.round(float(loss_all_epochs[-1]),5)}")
        print(f"Mean training accuracy: {np.round(float(train_accuracy_all_epochs[-1]),3)}")
        print(f"Mean confidence of predictions: {np.round(float(mean_confidence_all_epochs[-1]),3)}")
        
        
        # Calculate accuracy using test dataset
        if dataset_test:            
            for step, (xc, yc, xt, yt) in enumerate(dataset_test):
            
                if padding_values:
                    # A mask for where the padding is. This is the same shape as the batch
                    padding_mask = (yt == padding_values)
                    # This is collapsed along the categorical dimension
                    padding_mask = B.any(padding_mask,axis=-2)
                else:
                    padding_mask = None
                
                # Forward pass
                yt_pred, _, _, _ = nps.predict(
                    model,xc, yc, xt, num_samples=num_samples
                    ) 
                
                # Accuracy of the non-padding values for the test data
                accuracy=calc_cat_acc_onehot(yt,yt_pred,cat_axis=-2,padding_values=padding_mask)
                test_accuracy_per_epoch.update_state(accuracy)
            
            # Append to epoch metric
            test_accuracy_all_epochs.append(test_accuracy_per_epoch.result())
            print(f"Mean test accuracy: {np.round(float(test_accuracy_all_epochs[-1]),3)}")
        
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
        'train_accuracy' : train_accuracy_all_epochs,
        'cat_confidence' : mean_confidence_all_epochs
        }
    if dataset_test:
        history['test_accuracy'] = test_accuracy_all_epochs,
        
    return history

