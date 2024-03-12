# class RandomFourierFeatures
#  __init__
#       * initializes the random weights
#  __call__(x):
#       * maps the input x\in\bbR^d to the feature space R^D

#  - The mapping will depend on the kernel. Here are a couple of examples that will work:
#    * RBF kernel: sample iid Gaussians???? translation invariant kernel
#    * Linear kernel: sample iid Gaussians and then apply a polynomial function 


# class TRP_RFSF 
#     will have m independent classes of RandomFourierFeatures (slow...)



# FORGET THE ABOVE. probably need 2 different approaches for linear and RBF. START WITH RBF IMPLEMENTATION.



#i do however need a class for fourier features.
