# SPDX-FileCopyrightText: 2025 University of Manchester
#
# SPDX-License-Identifier: apache-2.0

import torch
import numpy as np
import math
import os
import lab as B
import pandas as pd
import neuralprocesses.torch as nps

from src.privacy_oracle import get_sigma_from_privacy_loss_distribution as get_sigma
from src.util import calc_greedy_confidence, calc_true_confidence, calc_greedy_acc_onehot, average_grads_batch_torch


class AverageMeter(object):
    """Computes and stores the average and current value. This is used as a 
    training metric.
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to zero."""
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        """Add a value."""
        self.sum += float(val)
        self.count += 1
        
    def result(self):
        """Return the average."""
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0


def train_model_dp_torch(
    model,
    dataloader_train,
    dataloader_train_metadata,
    loss_fn,
    optimizer,
    settings,
    dataloader_val=None    
):
    
    """
    Train a neural process model with differential privacy by streaming data
    from a torch dataloader. The model is modified in-place.

    Parameters
    ----------
    model : neuralprocesses.torch.Model
        The model to be trained.
    dataloader_train : torch.utils.data.DataLoader
        The dataloader to be used for training.
    dataloader_train_metadata : dict
        A dictionary with keys 'n_users', 'n_minibatches' representing the
        metadata of dataloader_train.
    loss_fn : callable
        The loss function to be used for training. Must be from the file
        dppum.loss
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    settings : dict
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
            Whether to shuffle dataloader_train before each epoch.
        models_dir : str
            The directory where the trained model will be saved.
        padding_value : float
            Padding value which will be discarded during the loss/accuracy calculations.
        clip_grads_per_user : str
            If set to 'false', model gradients will be calculated once per batch like normal.
            If set to 'loop', gradients will be calculated (and clipped if appropriate)
            for each user in the batch individually by looping through them.
    dataloader_val : torch.utils.data.DataLoader
        The dataloader to be used for validation during training.

    Returns
    -------
    history : dict
        A dictionary with the training metrics:
        epoch : int
            The training epochs.
        loss : float
            The training loss.
        train_acc_greedy : float
            The greedy accuracy tested on the training dataset. Take the most
            likely category from the neuralprocess model and see if its correct.
        train_acc_sample : float
            The sampled accuracy tested on the training dataset. Take the
            probability that the neuralprocess model gives for the true category.
        train_conf_greedy : float
            The confidence (probability) the neuralprocess model gives for the
            most likely category it predicts.
        val_acc_greedy : float
            The greedy accuracy tested on the validation dataset. Take the most
            likely category from the neuralprocess model and see if its correct.
            Only output if dataloader_val is provided.
        val_acc_sample : float
            The sampled accuracy tested on the validation dataset. Take the
            probability that the neuralprocess model gives for the true category.
            Only output if dataloader_val is provided.
        
    Notes
    -----
    The model is modified in-place so it is not returned.
    """
    
    # The number of times the gradients will be updated during training
    num_repeats = dataloader_train_metadata['n_batches'] * settings['num_epochs']
    # The fraction of the data that is used in each minibatch
    subsampling_rate = 1/dataloader_train_metadata['n_batches']
    # Calculate sigma
    sigma = get_sigma(settings['epsilon'], settings['delta'], num_repeats, subsampling_rate)

    # Define training metrics: loss, accuracy, and confidence of mean model category
    # These metrics are for within epochs, and are reset after each epoch
    train_acc_greedy_per_epoch = AverageMeter()
    train_acc_sample_per_epoch = AverageMeter()
    loss_per_epoch = AverageMeter()
    train_conf_greedy_per_epoch = AverageMeter()
    # Metrics to keep track of the encoder gradient L2 norms (useful to work out if c is too high/low)
    enc_norms_per_epoch = AverageMeter()
    dec_norms_per_epoch = AverageMeter()
    
    if dataloader_val:
        val_acc_greedy_per_epoch = AverageMeter()
        val_acc_sample_per_epoch = AverageMeter()  
        
    # Define lists to store the metrics for all epochs
    train_acc_greedy_all_epochs = []
    train_acc_sample_all_epochs = []
    loss_all_epochs = []
    train_conf_greedy_all_epochs = []    
    enc_norms_all_epochs = []
    dec_norms_all_epochs = []
    if dataloader_val:
        val_acc_greedy_all_epochs = []
        val_acc_sample_all_epochs = []
    
    
    # Use warmup epoch (or not)
    if settings['warmup_epoch']:
        first_epoch = 0
    else:
        first_epoch = 1        
        
    # Get the training device (normally GPU)
    training_device = get_device_type()


    def loss_wrapper(args_tuple, state=None):
        """
        A wrapper function to calculate the loss and gradients, clip the
        gradients (if required), and return the gradients.
        It is written in a way so it can be used in a vectorized map to 
        calculate and clip the gradients on a per-user basis efficiently
        
        args_tuple : tuple
        A tuple containing four PyTorch tensors (xc,yc,xt,yt) for the context
        and target data.
        state : B random state (optional).
        
        Returns
        -------
        encoder_gradients : list
            List of the encoder gradients.
        decoder_gradients : list
            List of the decoder gradients.
        
        """

        # Unwrap the input data
        xc,yc,xt,yt = args_tuple

        # If there's no batch dimension, add a batch dimension of 1
        if len(xc.shape) == 3:
            xc = B.expand_dims(xc, 0)
            yc = B.expand_dims(yc, 0)
            xt = B.expand_dims(xt, 0)
            yt = B.expand_dims(yt, 0)
        
        # Reset optimizer
        optimizer.zero_grad()
        
        # Compute the loss value for this minibatch.
        if not state:
            state = B.global_random_state(B.dtype(xc))
        
        vector_loss = -loss_fn(
                state=state,
                model=model,
                contexts=[(xc,yc)],
                subsume_context=True,
                xt=xt,
                yt=yt,
                normalise=False,
                dtype_lik=torch.float32,
                num_samples=settings['num_samples'],
                padding_value=settings['padding_value']
                )
        scalar_loss = B.mean(vector_loss)

        # Update loss metric
        loss_per_epoch.update(scalar_loss)
        
        # Backward pass
        scalar_loss.backward()

        # Get the gradients as tensors and calculate average L2 norms
        encoder_gradients = torch.cat([p.grad.view(-1) for p in model.encoder.parameters() if p.requires_grad])
        decoder_gradients = torch.cat([p.grad.view(-1) for p in model.decoder.parameters() if p.requires_grad])        
        
        enc_norm = torch.linalg.vector_norm(encoder_gradients, ord=2)
        dec_norm = torch.linalg.vector_norm(decoder_gradients, ord=2)
        enc_norms_per_epoch.update(enc_norm)
        dec_norms_per_epoch.update(dec_norm)
        
        # Get the gradients
        encoder_gradients = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        decoder_gradients = [p.grad for p in model.decoder.parameters() if p.requires_grad]
        
        if settings['dp_enc']:
            # Clip encoder gradients per user
            for i, grad in enumerate(encoder_gradients):
                if grad is not None:
                    # L2 clipping
                    grad_norm = torch.norm(grad)
                    grad_clipped = grad * min(1.0, settings['clipping_bound'] / grad_norm)
                    
                    # Store the clipped gradient
                    encoder_gradients[i] = grad_clipped.clone()
             
            # Add gaussian noise
            encoder_gradients = [tensor + torch.randn_like(tensor) * math.sqrt(settings['clipping_bound']**2 * sigma**2) for tensor in encoder_gradients]
                    
        if settings['dp_dec']:
            # Clip decoder gradients per user
            for i, grad in enumerate(decoder_gradients):
                if grad is not None:
                    # L2 clipping
                    grad_norm = torch.norm(grad)
                    grad_clipped = grad * min(1.0, settings['clipping_bound'] / grad_norm)
                    
                    # Store the clipped gradient
                    decoder_gradients[i] = grad_clipped.clone()
        
            decoder_gradients = [tensor + torch.randn_like(tensor) * math.sqrt(settings['clipping_bound']**2 * sigma**2) for tensor in decoder_gradients]
    
        return encoder_gradients, decoder_gradients
    
        
    print("Starting training loop")
    epochs = list(range(first_epoch, max(settings['num_epochs'], first_epoch) + 1))
    # Run the training loop
    for epoch in epochs:
        print(f"""######## Start of epoch {epoch} ########""")
        model.train()
        
        # Shuffle the training dataset
        if settings['shuffle']:
            dataloader_train = dataloader_train.shuffle(buffer_size=dataloader_train_metadata['n_batches'])
        
        # Reset epoch metrics
        train_acc_greedy_per_epoch.reset()
        train_acc_sample_per_epoch.reset()
        loss_per_epoch.reset()
        train_conf_greedy_per_epoch.reset()
        if dataloader_val:
            val_acc_greedy_per_epoch.reset()
            val_acc_sample_per_epoch.reset()

        # Iterate over the batches of the training dataset.
        for step, (xc, yc, xt, yt) in enumerate(dataloader_train):
            xc = xc.to(training_device)
            yc = yc.to(training_device)
            xt = xt.to(training_device)
            yt = yt.to(training_device)

            # First, calculate the loss/gradients
            # (the loss metric is updated within loss_wrapper so we don't need to see it explicitly here)
            if settings['clip_grads_per_user'] == 'loop':
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

                # Average the gradients over the batch   
                encoder_gradients = average_grads_batch_torch(encoder_gradients_batch)
                decoder_gradients = average_grads_batch_torch(decoder_gradients_batch)
            else:
                # Calculate and clip (if appropriate) gradients on a per-batch basis
                encoder_gradients, decoder_gradients = loss_wrapper((xc, yc, xt, yt))  
                
            # We have now calculated the loss/gradients
            # Apply gradients to update model
            # Update encoder parameters           
            for param, grad in zip(model.encoder.parameters(), encoder_gradients):
                param.grad = grad
                
            # Update decoder parameters
            for param, grad in zip(model.decoder.parameters(), decoder_gradients):
                param.grad = grad

            # If epoch is 0, that means we are doing a warmup and don't want
            # to apply the gradients
            if epoch > 0:
                optimizer.step()


        ##### End of epoch calculations #####
        model.eval()
        
        # Assess accuracy and confidence of predictions using training dataset
        for step, (xc, yc, xt, yt) in enumerate(dataloader_train):
            # Move tensors to training device (GPU)
            xc = xc.to(training_device)
            yc = yc.to(training_device)
            xt = xt.to(training_device)
            yt = yt.to(training_device)
            
            # Forward pass
            with torch.no_grad():
                yt_pred, _, _, _ = nps.predict(
                    model,xc, yc, xt, num_samples=settings['num_samples'], dtype_lik=torch.float32
                    ) 
            
            # Accuracy of the non-padding values
            greedy_accuracy = calc_greedy_acc_onehot(yt,yt_pred,cat_axis=-2,padding_value=settings['padding_value'])
            # Sample accuracy is the model's confidence in the true category
            sample_accuracy = calc_true_confidence(yt,yt_pred,-2,padding_value=settings['padding_value'])
            # Mean confidence of the non-padding values
            greedy_confidence = calc_greedy_confidence(yt_pred,-2,padding_value=settings['padding_value'])
            
            # Update accuracy and confidence metrics
            train_acc_greedy_per_epoch.update(greedy_accuracy)
            train_acc_sample_per_epoch.update(sample_accuracy)
            train_conf_greedy_per_epoch.update(greedy_confidence)
            

        # Append to epoch metrics
        loss_all_epochs.append(loss_per_epoch.result())
        train_acc_greedy_all_epochs.append(train_acc_greedy_per_epoch.result())
        train_acc_sample_all_epochs.append(train_acc_sample_per_epoch.result())
        train_conf_greedy_all_epochs.append(train_conf_greedy_per_epoch.result())
        
        # Print epoch metrics
        print(f"Loss: {np.round(float(loss_all_epochs[-1]),5)}")
        print(f"Greedy train accuracy: {np.round(float(train_acc_greedy_all_epochs[-1]),3)}")
        print(f"Sample train accuracy: {np.round(float(train_acc_sample_all_epochs[-1]),3)}")
        print(f"Greedy train confidence of predictions: {np.round(float(train_conf_greedy_all_epochs[-1]),3)}")
        
        # Metrics to keep track of the encoder gradient L2 norms (useful to work out if c is too high/low)
        enc_norms_all_epochs.append(enc_norms_per_epoch.result())
        dec_norms_all_epochs.append(dec_norms_per_epoch.result())
        
        
        # Calculate accuracy using validation dataset
        if dataloader_val:            
            # Assess accuracy and confidence of predictions using training dataset
            for step, (xc, yc, xt, yt) in enumerate(dataloader_val):
                # Move tensors to training device (GPU)
                xc = xc.to(training_device)
                yc = yc.to(training_device)
                xt = xt.to(training_device)
                yt = yt.to(training_device)
                
                # Forward pass
                with torch.no_grad():
                    yt_pred, _, _, _ = nps.predict(
                        model,xc, yc, xt, num_samples=settings['num_samples'], dtype_lik=torch.float32
                        ) 
                # Accuracy of the non-padding values
                greedy_accuracy = calc_greedy_acc_onehot(yt,yt_pred,cat_axis=-2,padding_value=settings['padding_value'])
                # Sample accuracy is the model's confidence in the true category
                sample_accuracy = calc_true_confidence(yt,yt_pred,-2,padding_value=settings['padding_value'])
                
                # Update accuracy and confidence metrics
                val_acc_greedy_per_epoch.update(greedy_accuracy)
                val_acc_sample_per_epoch.update(sample_accuracy)
            
            # Append to epoch metrics
            val_acc_greedy_all_epochs.append(val_acc_greedy_per_epoch.result())
            val_acc_sample_all_epochs.append(val_acc_sample_per_epoch.result())
            print(f"Greedy val accuracy: {np.round(float(val_acc_greedy_all_epochs[-1]),3)}")
            print(f"Sample val accuracy: {np.round(float(val_acc_sample_all_epochs[-1]),3)}")
        
        
        if settings['models_dir']:
            # Check if the directory exists
            if not os.path.exists(settings['models_dir']):
                # If not, create the directory
                os.makedirs(settings['models_dir'])
            
            # Save the model weights            
            model_name = f"weights_epoch_{epoch}.pt"
            torch.save(model.state_dict(), os.path.join(settings['models_dir'], model_name))

    
    ##### End of training loop #####
    
    # Return the history data (training metrics)
    history = {
        'epoch' : epochs,
        'loss' : loss_all_epochs,
        'train_acc_greedy' : train_acc_greedy_all_epochs,
        'train_acc_sample' : train_acc_sample_all_epochs,
        'train_conf_greedy' : train_conf_greedy_all_epochs,
        'enc_norms' : enc_norms_all_epochs,
        'dec_norms' : dec_norms_all_epochs
        
        }
    if dataloader_val:
        history['val_acc_greedy'] = val_acc_greedy_all_epochs
        history['val_acc_sample'] = val_acc_sample_all_epochs
    
    # Save the history to CSV
    if settings['models_dir']:
        # Convert the dictionary to a pandas DataFrame        
        csv_name = 'training_metrics.csv'
        csv_path = os.path.join(settings['models_dir'], csv_name)
        pd.DataFrame(history).round(3).set_index('epoch').to_csv(csv_path)
        
    
    return history



def get_device_type():
    """
    Returns the type of available GPU device:
    - "mps" (Metal Performance Shaders) for M1 GPUs
    - "cuda:x" for NVIDIA GPUs, where x is the index of the GPU
    - "cpu" if no GPU is available
    """
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        return f"cuda:{current_device}"
    else:
        return "cpu"

    

