from neuralprocesses.model.elbo import _merge_context_target, _kl
import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch
import torch.nn.functional as F

from dppum.util import reshape_to_last, swap_axes


def np_elbo_cat_torch(
    state: B.RandomState,
    model: nps.Model,
    contexts: list,
    xt,
    yt,
    *,
    cat_axis=-2,
    num_samples=1,
    normalise=False,
    subsume_context=False,
    fix_noise=None,
    dtype_lik=None,
    padding_values=None,
    **kw_args,
):
    """ELBO objective, with the log-likelihood part calculated using 
    tf.nn.softmax_cross_entropy_with_logits. Based on nps.elbo.
    As such it will only work with tensorflow tensors as the input data, with
    categorical y data. The output of this function should be the same as
    np_elbo_explicit, but this version is normally faster.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        cat_axis (int) : Categorical axis of y data. Defaults to -2.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.
        dtype_lik (dtype, optional): Data type to use for the likelihood computation.
            Defaults to the 64-bit variant of the data type of `yt`.
        padding_values : float, optional
            Padding value for yt which will be discarded during the loss calculations.

    Returns:
        random state, optional: Random state.
        tensor: ELBOs.
    """
    
    float = B.dtype_float(yt)
    float64 = B.promote_dtypes(float, np.float64)
    
    # For the likelihood computation, default to using a 64-bit version of the data
    # type of `yt`.
    if not dtype_lik:
        dtype_lik = float64

    if subsume_context:
        # Only here also update the targets.
        contexts_q, xt, yt = _merge_context_target(contexts, xt, yt)
    else:
        contexts_q, _, _ = _merge_context_target(contexts, xt, yt)

    # Construct prior.
    xz, pz, h = nps.code_track(
        model.encoder,
        *nps.util.compress_contexts(contexts),
        xt,
        root=True,
        dtype_lik=dtype_lik,
        **kw_args,
    )

    # Construct posterior.
    qz = nps.recode_stochastic(
        model.encoder,
        pz,
        *nps.util.compress_contexts(contexts_q),
        h,
        root=True,
        dtype_lik=dtype_lik,
        **kw_args,
    )

    # Sample from posterior.
    shape = () if num_samples is None else (num_samples,)
    state, z = nps.util.sample(state, qz, *shape)
    z = B.cast(float, z)
    
    # Run sample through decoder.
    _, d = nps.code(
        model.decoder,
        xz,
        z,
        xt,
        dtype_lik=dtype_lik,
        root=True,
        **kw_args,
    )
    d = nps.util.fix_noise(d, fix_noise)
    
    # d.mean is the mean of our samples from the latent distribution through 
    # the decoder
    # Transpose y_true and y_pred to shape [minibatch, num_categories,[data_dimensions]]
    # so they can go into torch.nn.CrossEntropyLoss correctly 
    
    yt_true_transposed = swap_axes(yt,cat_axis,1)
    yt_pred_transposed = swap_axes(d.mean[0], cat_axis,1)
    yt_pred_transposed = B.cast(dtype_lik,yt_pred_transposed)

    
    # Calculate the softmax cross-entropy reconstruction loss
    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    recon_loss = loss_function(yt_pred_transposed, yt_true_transposed)
    
    # If there is padding, make sure we set the reconstruction loss to zero
    if padding_values is not None:  # It gives an error if you do "if padding_values:"
        # If padding is a single value
        if B.size(padding_values) == 1:
            # Identify the padding
            padding_mask = (yt == padding_values)
            padding_mask = B.any(padding_mask, axis=cat_axis)
        else:
            raise ValueError("'padding_values' must be a single value")
            
        # For the padding parts, assign the loss to zero
        recon_loss = B.where(padding_mask, 0., recon_loss)    
    
    # Average loss over the number of target data points to match shape of _kl
    recon_loss = B.mean(recon_loss,axis=-1)
    
    # Calculate the ELBO loss
    # The KL loss is still useful for the parts that are padding
    elbos = -recon_loss - _kl(qz, pz)

    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(dtype_lik, nps.num_data(xt, yt))

    return elbos