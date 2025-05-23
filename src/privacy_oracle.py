# This script has been copied directly from the source repository
# https://github.com/hamalajaa/DifferentiablyPrivateProbabilisticUserModeling/blob/main/src/privacy_oracle.py

import scipy.optimize
import numpy as np
from dp_accounting.pld import privacy_loss_distribution

def get_sigma_basic(epsilon: float, delta: float, num_repeats: int, subsampling_rate: float=1.) -> float:
        """ Computes the required privacy noise level according to basic composition of privacy
        and privacy amplification by subsampling.
        Basic composition for L repititions: eps_total = L * eps_iter, delta_total = L * delta_iter
        Privacy amplification by subsampling: If any data point is included in the query with probability q,
        eps_subsampled = log(1 + q*(e^eps_basic - 1)) and delta_subsampled = q * delta_basic .
        Args:
        epsilon: Desired privacy parameter epsilon.
        delta: Desired privacy parameter delta.
        num_repeats: Number of repeated queries to the data (e.g., SGD iterations).
        subsampling_rate: The chance for any data point to be included in the query.
        """
        q = subsampling_rate

        if delta >= 1. or delta < 0:
            raise ValueError(f"delta must be a value between 0 and 1; got {delta}")
        if epsilon <= 0:                        
            raise ValueError(f"epsilon must be a positive value; got {epsilon}")    
        if q > 1. or q < 0.:
            raise ValueError(f"subsampling_rate must be a value between 0 and 1; got {subsampling_rate}")
    
        iter_epsilon = np.log(1. + (np.exp( epsilon / num_repeats ) - 1.) / q)
        iter_delta = delta / (q * num_repeats)
        return np.sqrt(2 * np.log(1.25 / iter_delta)) / iter_epsilon

def get_sigma_from_privacy_loss_distribution(
            epsilon: float, delta: float, num_repeats: int, subsampling_rate: float=1., 
            xtol: float=2e-12) -> float:
        """ Computes the required privacy noise level using the privacy loss distribution.
            
            Args:
                 epsilon: Desired privacy parameter epsilon.
                 delta: Desired privacy parameter delta.                                       
                 num_repeats: Number of repeated queries to the data (e.g., SGD iterations).
                 subsampling_rate: The chance for any data point to be included in the query.
        """
        
        def delta_err(sigma):
            pld = privacy_loss_distribution.from_gaussian_mechanism(sigma, sampling_prob=subsampling_rate).self_compose(num_repeats)
            return pld.get_delta_for_epsilon(epsilon) - (delta - xtol)
                                                                                    
        def delta_(sigma):
            pld = privacy_loss_distribution.from_gaussian_mechanism(sigma, sampling_prob=subsampling_rate).self_compose(num_repeats)
            return pld.get_delta_for_epsilon(epsilon)
                                                                                                        
        sigma_upper = get_sigma_basic(epsilon, delta, num_repeats, subsampling_rate)                   
        
        #print(f"{sigma_upper=}", file=sys.stderr)
        #print("{}".format(delta_(sigma_upper)), file=sys.stderr)
        #print(delta_err(sigma_upper), file=sys.stderr)
        
        assert delta_err(sigma_upper) < 0.
        
        # while delta_err(sigma_upper) > 0.:
        #     sigma_upper *= 1.5                                
        sigma_lower = np.sqrt(sigma_upper)                                                                                                                               
        while delta_err(sigma_lower ) < 0.:                              
            sigma_lower = .5 * sigma_lower
        
        return scipy.optimize.brentq(delta_err, sigma_lower, sigma_upper)

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        "Privacy oracle script: Given DP-SGD parameters and desired privacy levels, prints the required noise level sigma for a single iteration to stdout."    
    )
    parser.add_argument("epsilon", type=float, help="Desired privacy parameter epsilon")
    parser.add_argument("delta", type=float, help="Desired privacy parameter delta.")
    parser.add_argument("num_repeats", type=int, help="Number of SGD iterations.")
    parser.add_argument("--subsampling_rate", default=1., type=float, help="The chance for any data point to be included in the query.")
    args = parser.parse_args()
    #print(args, file=sys.stderr)

    #try:
    print(get_sigma_from_privacy_loss_distribution(**(args.__dict__)))    
    #except Exception as e:                                                                  
    #    print(e, file=sys.stderr)
    #    exit(1)