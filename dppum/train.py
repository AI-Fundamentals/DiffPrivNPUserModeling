import torch
import numpy as np
import json
import os
import lab as B
import pandas as pd
import neuralprocesses.torch as nps

from dppum.privacy_oracle import get_sigma_from_privacy_loss_distribution as get_sigma
from dppum.util import calc_cat_confidence, flatten_first_two_dims, calc_cat_acc_onehot
  
    
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
    dataset_train,
    dataset_train_metadata,
    loss_fn,
    optimizer,
    settings,
    dataset_val=None    
):
    
    """
    Train a neural process model with differential privacy by streaming data
    from a torch dataloader. The model is modified in-place.

    Parameters
    ----------
    model : neuralprocesses.torch.Model
        The model to be trained.
    dataset_train : torch.utils.data.DataLoader
        The dataset to be used for training.
    dataset_train_metadata : dict
        A dictionary with keys 'n_users', 'n_minibatches' representing the
        metadata of dataset_train.
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
            Whether to shuffle dataset_train before each epoch.
        models_dir : str
            The directory where the trained model will be saved.
        padding_value : float
            Padding value which will be discarded during the loss/accuracy calculations.
        clip_grads_per_user : str
            If set to 'false', model gradients will be calculated once per batch like normal.
            If set to 'loop', gradients will be calculated (and clipped if appropriate)
            for each user in the batch individually by looping through them.
    dataset_val : torch.utils.data.DataLoader
        The dataset to be used for validation during training.

    Returns
    -------
    history : dict
        A dictionary with the training metrics: loss, accuracy, confidence.

    Notes
    -----
    The model is modified in-place so it is not returned.
    """
    
    # The number of times the gradients will be updated during training
    num_repeats = dataset_train_metadata['n_batches'] * settings['num_epochs']
    # The fraction of the data that is used in each minibatch
    subsampling_rate = 1/dataset_train_metadata['n_batches']
    # Calculate sigma
    sigma = get_sigma(settings['epsilon'], settings['delta'], num_repeats, subsampling_rate)

    # Define training metrics: loss, accuracy, and confidence of mean model category
    # These metrics are for within epochs, and are reset after each epoch
    train_accuracy_per_epoch = AverageMeter()
    loss_per_epoch = AverageMeter()
    mean_confidence_per_epoch = AverageMeter()
    if dataset_val:
        val_accuracy_per_epoch = AverageMeter()
    
    # Define lists to store the metrics for all epochs
    train_accuracy_all_epochs = []
    loss_all_epochs = []
    mean_confidence_all_epochs = []    
    if dataset_val:
        val_accuracy_all_epochs = []
    
    # Use warmup epoch (or not)
    if settings['warmup_epoch']:
        first_epoch = 0
    else:
        first_epoch = 1        
        
    # Get the training device (normally GPU)
    training_device = get_device_type()


    def loss_wrapper(args_tuple):
        """
        A wrapper function to calculate the loss and gradients, clip the
        gradients (if required), and return the gradients.
        It is written in a way so it can be used in a vectorized map to 
        calculate and clip the gradients on a per-user basis efficiently
        
        args_tuple : tuple
        A tuple containing four PyTorch tensors (xc,yc,xt,yt) for the context
        and target data.
        
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
        
        # Get the gradients
        encoder_gradients = [p.grad for p in model.encoder.parameters() if p.requires_grad]
        decoder_gradients = [p.grad for p in model.decoder.parameters() if p.requires_grad]
        
        if settings['dp_enc']:
                # Clip encoder gradients per user
            for i, grad in enumerate(encoder_gradients):
                if grad is not None:
                    # L2 clipping
                    grad_norm = torch.norm(grad)
                    grad_normalized = grad / grad_norm
                    grad_clipped = grad_normalized * min(1.0, settings['clipping_bound'] / grad_norm)
                    
                    # Store the clipped gradient
                    encoder_gradients[i] = grad_clipped.clone()
        if settings['dp_dec']:
                # Clip decoder gradients per user
                for i, grad in enumerate(decoder_gradients):
                    if grad is not None:
                        # L2 clipping
                        grad_norm = torch.norm(grad)
                        grad_normalized = grad / grad_norm
                        grad_clipped = grad_normalized * min(1.0, settings['clipping_bound'] / grad_norm)
                        
                        # Store the clipped gradient
                        decoder_gradients[i] = grad_clipped.clone()
        
        return encoder_gradients, decoder_gradients
    
        
    print("Starting training loop")
    model.train()
    epochs = list(range(first_epoch, max(settings['num_epochs'], first_epoch) + 1))
    # Run the training loop
    for epoch in epochs:
        print(f"""######## Start of epoch {epoch} ########""")
        
        # Shuffle the training dataset
        if settings['shuffle']:
            dataset_train = dataset_train.shuffle(buffer_size=dataset_train_metadata['n_batches'])
        
        # Reset epoch metrics
        train_accuracy_per_epoch.reset()
        loss_per_epoch.reset()
        mean_confidence_per_epoch.reset()
        if dataset_val:
            val_accuracy_per_epoch.reset()

        # Iterate over the batches of the training dataset.
        for step, (xc, yc, xt, yt) in enumerate(dataset_train):

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
            
            # Add gaussian noise
            if settings['dp_enc']:
                encoder_gradients = [tensor + torch.randn_like(tensor) * (settings['clipping_bound']**2 * sigma**2) for tensor in encoder_gradients]
            if settings['dp_dec']:
                decoder_gradients = [tensor + torch.randn_like(tensor) * (settings['clipping_bound']**2 * sigma**2) for tensor in decoder_gradients]
            
            
            # We have now calculated the loss/gradients
            # Apply gradients to update model
            # Update encoder parameters           
            for param, grad in zip(model.encoder.parameters(), encoder_gradients):
                #param.data.sub_(optimizer.param_groups[0]['lr'] * grad)
                param.grad = grad
                
            # Update decoder parameters
            for param, grad in zip(model.decoder.parameters(), decoder_gradients):
                #param.data.sub_(optimizer.param_groups[0]['lr'] * grad)
                param.grad = grad

            # If epoch is 0, that means we are doing a warmup and don't want
            # to apply the gradients
            if epoch > 0:
                optimizer.step()

            # Assess accuracy after updating model weights
            with torch.no_grad():
                yt_pred, _, _, _ = nps.predict(
                    model,xc, yc, xt, num_samples=settings['num_samples'], dtype_lik=torch.float32
                    )
            
            # Create mask for the padding
            if settings['padding_value']:
                # A mask for where the padding is. This is the same shape as the batch
                padding_mask = (yt == settings['padding_value'])
                # This is collapsed along the categorical dimension
                padding_mask = B.any(padding_mask,axis=-2)
            else:
                padding_mask = None
            
            # Accuracy of the non-padding values
            accuracy=calc_cat_acc_onehot(yt,yt_pred,cat_axis=-2,padding_value=padding_mask)
            # Mean confidence of the non-padding values
            confidence = calc_cat_confidence(yt_pred,-2,padding_mask)
            
            # Update accuracy and confidence metrics
            train_accuracy_per_epoch.update(accuracy)
            mean_confidence_per_epoch.update(confidence)


        ##### End of epoch calculations #####

        # Append to epoch metrics
        loss_all_epochs.append(loss_per_epoch.result())
        train_accuracy_all_epochs.append(train_accuracy_per_epoch.result())
        mean_confidence_all_epochs.append(mean_confidence_per_epoch.result())
        
        # Print epoch metrics
        print(f"Loss: {np.round(float(loss_all_epochs[-1]),5)}")
        print(f"Mean training accuracy: {np.round(float(train_accuracy_all_epochs[-1]),3)}")
        print(f"Mean confidence of predictions: {np.round(float(mean_confidence_all_epochs[-1]),3)}")
        
        
        # Calculate accuracy using validation dataset
        if dataset_val:            
            for step, (xc, yc, xt, yt) in enumerate(dataset_val):
                # Move tensors to training device (GPU)
                xc = xc.to(training_device)
                yc = yc.to(training_device)
                xt = xt.to(training_device)
                yt = yt.to(training_device)
                
                if settings['padding_value']:
                    # A mask for where the padding is. This is the same shape as the batch
                    padding_mask = (yt == settings['padding_value'])
                    # This is collapsed along the categorical dimension
                    padding_mask = B.any(padding_mask,axis=-2)
                else:
                    padding_mask = None
                
                # Forward pass
                with torch.no_grad():
                    yt_pred, _, _, _ = nps.predict(
                        model,xc, yc, xt, num_samples=settings['num_samples'], dtype_lik=torch.float32
                        ) 
                
                # Accuracy of the non-padding values for the validation data
                accuracy=calc_cat_acc_onehot(yt,yt_pred,cat_axis=-2,padding_value=padding_mask)
                val_accuracy_per_epoch.update(accuracy)
            
            # Append to epoch metric
            val_accuracy_all_epochs.append(val_accuracy_per_epoch.result())
            print(f"Mean val accuracy: {np.round(float(val_accuracy_all_epochs[-1]),3)}")
        
        
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
        'loss' : loss_all_epochs,
        'train_accuracy' : train_accuracy_all_epochs,
        'cat_confidence' : mean_confidence_all_epochs,
        'epoch' : epochs
        }
    if dataset_val:
        history['val_accuracy'] = val_accuracy_all_epochs
    
    
    # Save the history to CSV
    if settings['models_dir']:
        # Convert the dictionary to a pandas DataFrame        
        csv_name = 'training_metrics.csv'
        csv_path = os.path.join(settings['models_dir'], csv_name)
        pd.DataFrame(history).set_index('epoch').to_csv(csv_path)
        
    
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
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            return f"cuda:{current_device}"
        else:
            return "cpu"

    

def average_grads_batch_torch(tensor_list):
    """
    Compute the average of tensors over the batch dimension. This is designed
    to average a list of lists of gradients into one single list of gradients.

    Parameters
    ----------
    tensor_list : list of list of torch.Tensor
        The input list of lists of tensors. The dimensions of this list of
        lists are [batch,num_tensors,tensor_sizes].

    Returns
    -------
    list of torch.Tensor
        The output list of averaged tensors. The dimensions of this list are
        [num_tensors,tensor_sizes].

    """
    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the number of tensors
    for i in range(len(tensor_list[0])):
        # Initialize a list to store the tensors of the current index from each batch
        tensors = []

        # Iterate over the batches
        for j in range(len(tensor_list)):
            # Append the tensor of the current index from the current batch to the list
            tensors.append(tensor_list[j][i])

        # Convert the list of tensors to a 3D tensor
        tensors = torch.stack(tensors)

        # Calculate the mean over the batch dimension (0th dimension) and
        # append the result to the averaged tensors list
        averaged_tensors.append(torch.mean(tensors, dim=0))

    # Return the list of averaged tensors
    return averaged_tensors