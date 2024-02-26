# A basic example to check if you can run a basic GNP model from the
# neuralprocesses library
# Example from https://wessel.ai/neuralprocesses/basic_usage.html
# Tensorflow version

try:
    import lab as B
    import tensorflow as tf
    import neuralprocesses.tensorflow as nps    

    cnp = nps.construct_gnp(dim_x=2, dim_y=3, likelihood="lowrank")
    dist = cnp(
        B.randn(tf.float32, 16, 2, 10),  # Context inputs
        B.randn(tf.float32, 16, 3, 10),  # Context outputs
        B.randn(tf.float32, 16, 2, 15),  # Target inputs
    )
    mean, var = dist.mean, dist.var  # Prediction for target outputs

    print(dist.logpdf(B.randn(tf.float32, 16, 3, 15)))
    print(dist.sample())
    print(dist.kl(dist))
    print(dist.entropy())

except Exception as e:
    print(f"An error occurred: {e}")
