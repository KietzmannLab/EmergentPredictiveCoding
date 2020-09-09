# EmergentPredictiveCoding

Code and models accompanying my thesis project "Predictive Coding as an Emergent Phenomenon".

In this thesis I use two network architectures to show that two components of predictive coding, namely error calculation units and prediction units, can emerge in certain conditions without being explicitely modelled.

The first architecture is inspired by the PredNet model by [Lotter et al. (2016)](https://arxiv.org/abs/1605.08104), but simplified in a number of ways (one module, fully connected) and trained on a different objective.

The second architecture is even less restrictive with respect to modelling assumptions, since every unit is wired in the same way. This model is dubbed the 'Bathtub' model. The state `h` of the bathtub units is calculated as follows:


```
h(t) = ReLU[x(t) + Wh(t-1)]
```

where `W` is the weight matrix and `x` the input vector.

The networks are trained to minimise the 'presynaptic' activity of the neurons, i.e. the unit state before applying the activation function. The rationale is that this minimises neural activity, while preventing over-inhibition. It is akin to the method of excitatory-inhibitory balance used by [Denève et al. (2017)](https://doi.org/10.1016/j.neuron.2017.05.016).

The notebook [Thesis plots.ipynb](https://github.com/elgar-groot/EmergentPredictiveCoding/blob/master/Thesis%20plots.ipynb) contains the code used to generate the thesis figures.

