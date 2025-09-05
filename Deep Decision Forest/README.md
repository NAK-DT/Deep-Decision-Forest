
Node.py – Defines the basic node structure used in decision trees. Each node stores split conditions, children, and leaf predictions. It serves as the building block for the custom decision tree implementation.

Trees.py – Implements the DecisionTree and higher-level tree operations used in DDF. Handles training, prediction, and layer-wise construction of trees. This is where single-tree logic comes together into ensembles.

backpropagation.py – Implements the backpropagation-inspired retraining strategies. Instead of gradient descent, this module selects features (via Shapley values or exhaustive search) and retrains trees across layers to refine performance. The current heuristics that are available are SU and averageNode Depth, simply uncomment one of them in methods find_best_improvement_prop and find_best_improvement_single.

modify_data.py – Provides data preprocessing and transformation utilities. Includes train/test splitting, normalization, encoding, and preparing outputs between layers in the DDF pipeline.

redundancycheck.py – Contains methods to quantify redundancy in datasets using correlation- and PCA-based metrics. Produces the final redundancy score that motivates the use of DDF in redundancy-heavy datasets.

strategy.py – Defines training strategies for different modes of tree construction (e.g., random, greedy, or restricted). Experimenting with different decision rules inside the DDF architecture such as choosing heuristics found in backpropagation.py.

Examples of runs can be found in the folder above. 

Visualization of the DDF's trees can be found in the folder above. 
