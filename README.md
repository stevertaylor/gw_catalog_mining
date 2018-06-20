# gw_catalog_mining
Code and examples for "Mining Gravitational-wave Catalogs To Understand Binary Stellar Evolution: A New Hierarchical Bayesian Framework" by [Stephen  Taylor](https://github.com/stevertaylor) and [Davide Gerosa](https://github.com/dgerosa).

If you use this code for your research, please cite this paper.



Users can open the included `jupyter` notebook using by typing `jupyter notebook results_toymodel.ipynb` in a terminal. Individual cells can be executed by pressing `Shift + Tab`.

## Requirements

### Core

The following 4 packages are core, and can be downloaded as a bundle with [Anaconda](https://www.anaconda.com/distribution).

* python
* jupyter
* numpy
* scipy

### Essential

* [pyDOE](https://pythonhosted.org/pyDOE)
* [astropy](http://www.astropy.org)
* [gwdet](https://github.com/dgerosa/gwdet)
* [george](https://george.readthedocs.io/en/latest)

### Optional

* [emcee](http://dfm.io/emcee/current/) [for sampling]
* [chainconsumer](https://samreay.github.io/ChainConsumer) [for plotting marginalized posterior distributions]
* [tensorly](http://tensorly.org/stable/index.html) [for multi-linear PCA tests]