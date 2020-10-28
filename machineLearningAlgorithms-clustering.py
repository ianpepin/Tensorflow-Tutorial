import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions     # making a shortcut for later on
initialDistribution = tfd.Categorical(probs=[0.8, 0.2])     # refers to point 2
transitionDistributions = tfd.Categorical(probs=[[0.7
                                                     , 0.3], [0.2, 0.8]])       # refers to point 3
observationDistribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])        # refers to point 5
# the loc argument represents the mean and the scale is the standard deviation

# Create model
model = tfd.HiddenMarkovModel(
    initial_distribution=initialDistribution,
    transition_distribution=transitionDistributions,
    observation_distribution=observationDistribution,
    num_steps=7
)
mean = model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
# due to the wat tensorflow works on a lower level, we need to evaluate part of the graph from within a session to see
# the value of this tensor
