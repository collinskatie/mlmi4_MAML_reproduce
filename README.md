# mlmi4_MAML_reproduce
Reproducing MAML for MLMI 4 class project

Notebooks are employed for each main experiment and model. Sine regression domain results largely work from the data.py, models.py, utils.py, and analysis_utils.py functions, respectively. Constants contain primary hyperparameters, which are only changed for individual algorithmic comparisons and experiments as needed, and specified in the report. GNN_notebooks_and_pretrained_models contains code run for GNNs and ensembling experiments. 

Additional specific experiments, for instance, over Graph Neural Networks, multi-dimensional sines, and Mixture of Gaussians have dedicated folders with associated notebooks and sets of helper functions. The archive folder contains older editions of notebooks, which were largely for debugging and experimenation purposes. Note, the Reptile directory contains older code when we were experimenting with different varieties of Reptile, but does not contain notebooks used for the final report. 

Sources of inspiration, or other codebases and/or sources from which code is pulled or heavily modified from, is noted in the associated comments. 

Note, seeds are largely employed throughout. This is both for reproducibility and to enable us to extract representative results for discussion. As such, they are not completely random -- and can be re-run with different settings to visualize the diversity of algorithmic behavior. However, wherever possible, we endeveoured to depict typical performance. 

Additional note for transparency -- shortly before the deadline, we discovered a bug in our confidence interval (95% CI) code used to generate 95% CI bands for our k-shot evaluation table results, and confidence bars in figures. To patch this error, a remediation function was employed: 

import numpy as np
def reverse(x): 
  return np.sqrt(x*np.sqrt(6)/1.96)*1.96/10
  
This code was able to reverse-engineer the proper confidence bars, as we used the same settings per comparison. If needed, code can be re-run in full; we used the patch at the end as we did not have time to re-train and evaluate all models -- mean performance was unchanged in this patch.


