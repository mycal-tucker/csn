## Concept Subspace Network

Code for the concept subspace network (CSN) as introduced by Tucker et al. in "Prototype Based Classification from Hierarchy to Fairness" in ICML 2022.

This is a keras-based implementation; creating a pytorch-based implementation should be relatively straightforward.

The key functionality is in proto_model.py, in which we define the CSN model.
There are lots of helper methods associated with the class for things like visualizing the latent space, calculating orthogonal subspaces, and more.

To get started, install all the requirements in ``requirements.txt``, with a python 3.6 interpreter.
Then, for a basic test of functionality run ``src/script/train_mnist.py``. You'll see lots of the parameters defined in that simple script, like alignment losses.

## Basic scripts

Many of the interesting scripts are included in ``src/scripts.``

Each script is relatively short: it defines many of the relevant parameters for training a CSN like the number of prototypes, relevant weights for different losses, etc.
Then the rest of the script is just a loop over random seeds in which a model is trained and then evaluated.

For example, in ``train_cifar100_deep.py``, we load the CIFAR100 data, set all the disentanglement weights and more, and then train the model.
The key datasets from our paper, excluding the bolt-placement task, which includes human-study data, are the German and Adult datasets, and CIFAR10 and CIFAR100.

## Overall Model

The key class to understand, in which we define the CSN, is in ``proto_model.py``.

Basically, the constructor defines various subparts, like an encoder, the prototypes, and a decoder (if we want to visualize the prototypes).
You'll see lots of different methods defining different architectures for these parts (e.g., a ConvNet for CIFAR10, or just dense layers for MNIST).

These parts all get put together in ``self.build_overall_network()`` which wires together the outputs of the encoder to get distances to prototypes, to generate predictions (as explained in the CSN).
We also calculate the alignment between the concept subspaces in this method and compute alignment losses.

That's it. There are lots of other helper methods for evaluation of visualization. I will perhaps release a pytorch version of this code that's much simpler and cleaner, so I haven't bothered cleaning up all these methods.

## Citation

If you found this useful, please cite the ICML paper (Bibtex to be included when ICML is officially published).

And feel free to reach out with questions to mycal@mit.edu
