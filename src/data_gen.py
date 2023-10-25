"""
Implementations for the data generators used in the experiments.
Currently implemented in tensorflow. Ideally these would be implemented in lab?
"""

import tensorflow_probability as tfp

class SearchEnvSampler:
    """
    This is producting abstract distributions that you can sample from
    """
    
    def __init__(self, args, menu_recall_probability=None, focus_duration_100ms=None, selection_delay_s=None):
        self.args = args
        self.menu_recall_probability = menu_recall_probability if menu_recall_probability is not None else tfp.distributions.Beta(3.0, 1.35)
        self.focus_duration_100ms = focus_duration_100ms if focus_duration_100ms is not None else tfp.distributions.TruncatedNormal(3.0, 1.0, 0.0, 5.0)
        self.selection_delay_s = selection_delay_s if selection_delay_s is not None else tfp.distributions.TruncatedNormal(0.3, 0.3, 0.0, 1.0)

    def __call__(self, x, noise):
        #This currently doesn't exist in python so will give an error
        return NeuralProcesses.FDD(x, noise, self)

