from neuralprocesses.model.elbo import _merge_context_target, _kl
import lab as B
import neuralprocesses.tensorflow as nps
import numpy as np
import tensorflow as tf

from neuralprocesses import normal

def np_elbo_explicit(
    state: B.RandomState,
    model: nps.Model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    normalise=False,
    subsume_context=False,
    fix_noise=None,
    dtype_lik=None,
    **kw_args,
):
    """ELBO objective, calculated explicitly.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.
        dtype_lik (dtype, optional): Data type to use for the likelihood computation.
            Defaults to the 64-bit variant of the data type of `yt`.

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
    
    #import pdb
    #pdb.set_trace()
    
    cat_axis = 1
    # We have already sampled from the latent distribution
    # Transpose y_true and y_pred to shape [minibatch, num_data_points, num_categories]
    # So they can go into softmax_cross_entropy_with_logits correctly 
    #yt_true_transposed = B.transpose(yt, perm=[0,2,1])
    yt_pred = d.mean[0]
    #yt_pred_transposed = B.transpose(d.mean[0], perm=[0,2,1])
    yt_pred = B.cast(dtype_lik,yt_pred)

    
    # Compute the loglik using Alex's method
    # Calculate the probabilities of the different categories by applying a softmax
    yt_pred_prob = B.softmax(yt_pred,axis=cat_axis)
    
    log_loss = logpdf_explicit(yt_pred_prob, yt,axis=cat_axis)
    #log_loss = tf.reduce_mean(log_loss,axis=[0])
    # Average loss over the data samples/tasks
    log_loss = tf.reduce_mean(log_loss,axis=[-1])
    
    elbos = log_loss - _kl(qz, pz)


    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(dtype_lik, nps.num_data(xt, yt))

    return elbos




def elbo_tf_cat(
    state: B.RandomState,
    model: nps.Model,
    contexts: list,
    xt,
    yt,
    *,
    num_samples=1,
    normalise=False,
    subsume_context=False,
    fix_noise=None,
    dtype_lik=None,
    **kw_args,
):
    """ELBO objective.

    Args:
        state (random state, optional): Random state.
        model (:class:`.Model`): Model.
        xc (input): Inputs of the context set.
        yc (tensor): Output of the context set.
        xt (input): Inputs of the target set.
        yt (tensor): Outputs of the target set.
        num_samples (int, optional): Number of samples. Defaults to 1.
        normalise (bool, optional): Normalise the objective by the number of targets.
            Defaults to `False`.
        subsume_context (bool, optional): Subsume the context set into the target set.
            Defaults to `False`.
        fix_noise (float, optional): Fix the likelihood variance to this value.
        dtype_lik (dtype, optional): Data type to use for the likelihood computation.
            Defaults to the 64-bit variant of the data type of `yt`.

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
    
    #import pdb
    #pdb.set_trace()
    
  

    #VERSION THAT WORKS WITH THE MEAN
    # We have already sampled from the latent distribution
    # Transpose y_true and y_pred to shape [minibatch, num_data_points, num_categories]
    # So they can go into softmax_cross_entropy_with_logits correctly 
    yt_true_transposed = B.transpose(yt, perm=[0,2,1])
    yt_pred_transposed = B.transpose(d.mean[0], perm=[0,2,1])
    yt_pred_transposed = B.cast(dtype_lik,yt_pred_transposed)

    # # VERSION THAT SAMPLES
    # # Sample yt_pred and transpose
    # yt_pred_transposed = tf.transpose(d.sample(),perm=[0,1,3,2])
    
    # # Transpose yt_true so it's right for the loss function
    # yt_true_transposed = tf.transpose(yt, perm=[0,2,1])
    # # Now give it an extra dimension to match the sampled predictions
    # yt_true_transposed = tf.expand_dims(yt_true_transposed, axis=0)
    # if yt_pred_transposed.shape[0] >1:
    #     yt_true_transposed = tf.tile(yt_true_transposed, [num_samples, 1, 1, 1])
    # yt_true_transposed = B.cast(dtype_lik,yt_true_transposed)
    

    #Reconstruction loss
    #Categorical crossentropy
    recon_loss = tf.nn.softmax_cross_entropy_with_logits(labels=yt_true_transposed, logits=yt_pred_transposed)
    # Log-likelihood
    #recon_loss = -tf.reduce_sum(yt_true_transposed * tf.math.log(yt_pred_transposed), axis=-1)
    # Average loss over the model samples
    recon_loss = -tf.reduce_mean(recon_loss,axis=[0])
    # Average loss over the data samples/tasks
    #recon_loss = tf.reduce_mean(recon_loss,axis=[-1])
    
    # # # Compute the ELBO using my loss
    # elbos = - (recon_loss + _kl(qz, pz))
    
    
    # Compute the ELBO using Wessel's loglik loss
    #elbos = B.mean(d.logpdf(B.cast(dtype_lik, yt)), axis=0) - _kl(qz, pz)
    
    #import pdb
    #pdb.set_trace()
    
    # Compute the loglik using Alex's method
    # Calculate the probabilities of the different categories by applying a softmax
    yt_pred_prob = B.softmax(yt_pred_transposed,axis=-1)
    
    log_loss = logpdf_explicit(yt_pred_prob, yt_true_transposed)
    #log_loss = tf.reduce_mean(log_loss,axis=[0])
    # Average loss over the data samples/tasks
    log_loss = tf.reduce_mean(log_loss,axis=[-1])
    
    elbos = log_loss - _kl(qz, pz)
    
    print("REcon_loss:", recon_loss[0])
    print("Log_loss:", log_loss[0])


    if normalise:
        # Normalise by the number of targets.
        elbos = elbos / B.cast(dtype_lik, nps.num_data(xt, yt))

    return elbos




def logpdf_explicit(d, x, axis=-1):
    """
    Explicitly compute the natural logarithm of the maximum product along each
    on the given axis

    Parameters
    ----------
    d : tensor-like
        The first input tensor.
    x : tensor-like
        The second input tensor.
    axis : int
        The axis along which to compute the calculation. Default is -1.

    Returns
    -------
    tf.Tensor
        A tensor representing the natural logarithm of the maximum product along each row.

    """
    return B.log(B.max(x * d, axis=axis))
