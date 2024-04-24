import neuralprocesses.torch as nps
from dppum.loss import np_elbo_explicit, np_elbo_cat_torch



# A test where you check that the two loss functions give the same thing
# for the same data, and random state I guess