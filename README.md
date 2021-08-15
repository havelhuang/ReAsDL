# ReAsDL
The Reliability Assessment Model (RAM) for Deep Learning Classifiers, presented at the AISafety'21 workshop co-located with IJCAI'21 (here is a [video](https://youtu.be/yb8DQokY3e8) presentation). Ths repository contains the illustrative examples for three synthesized datasets, MNIST dataset, and CIFAR10 dataset used in the [preprint](https://arxiv.org/abs/2106.01258).
## Models trained on synthesized datasets
we train random forests on systhesized datasets and apply RAM method to evluate the reliability. To running example fold and run
```
python main
```
## Deep learning models trained on MNIST and CIFAR10
we project the high dimensional inputs (images) into the latent space via Variantional Auto-Encoder (VAE), then apply RAM method get the final evaluation results. Type the command
```
python main('mnist', 'before', cell_size = 100, count_mh_steps = 250, count_particles = 10000)
```
