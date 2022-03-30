# mlmi4_MAML_reproduce
Reproducing MAML for MLMI 4 class project

Notebooks are employed for each main experiment and model. Sine regression domain results largely work from the data.py, models.py, utils.py, and analysis_utils.py functions, respectively. Constants contain primary hyperparameters, which are only changed for individual algorithmic comparisons and experiments as needed, and specified in the report. 

Additional specific experiments, for instance, over Graph Neural Networks, multi-dimensional sines, and Mixture of Gaussians have dedicated folders with associated notebooks and sets of helper functions. The archive folder contains older editions of notebooks, which were largely for debugging and experimenation purposes. 

Sources of inspiration, or other codebases and/or sources from which code is pulled or heavily modified from, is noted in the associated comments. 

Note, seeds are largely employed throughout. This is both for reproducibility and to enable us to extract representative results for discussion. As such, they are not completely random -- and can be re-run with different settings to visualize the diversity of algorithmic behavior. However, wherever possible, we endeveoured to depict typical performance. 


