
Node.py – Defines the basic node structure used in decision trees. Each node stores split conditions, children, and leaf predictions. It serves as the building block for the custom decision tree implementation.

Trees.py – Implements the DecisionTree and higher-level tree operations used in DDF. Handles training, prediction, and layer-wise construction of trees. This is where single-tree logic comes together into ensembles.

backpropagation.py – Implements the backpropagation-inspired retraining strategies. Instead of gradient descent, this module selects features (via Shapley values or exhaustive search) and retrains trees across layers to refine performance. The current heuristics that are available are SU and averageNode Depth, simply uncomment one of them in methods find_best_improvement_prop and find_best_improvement_single.

modify_data.py – Provides data preprocessing and transformation utilities that the algorithm uses. Includes train/test splitting, normalization, encoding, and preparing outputs between layers in the DDF pipeline.

redundancycheck.py – Contains methods to quantify redundancy in datasets using correlation- and PCA-based metrics. Produces the final redundancy score that motivates the use of DDF in redundancy-heavy datasets.

strategy.py – Defines training strategies for different modes of tree construction (e.g., random, greedy, or restricted). Experimenting with different decision rules inside the DDF architecture such as choosing heuristics found in backpropagation.py.

Visualization of the DDF is still in process, but current implementation can be found in Visualization. The idea is that by generating optimal decision trees from the package PYDL8.5, you can then compare an optimal single tree to the model that DDF actually learns. Theoretically speaking, by using hierarchical features DDF should converge to capture the present redundant subtrees from the optimal single tree in the higher layers of the DDF model. 

Examples of runs can be found in the folder above. 


#A thorough guide on how to run DDF:

#1: Prepare any dataset along with its preprocessing procedure. DDF works with NumPy arrays, so make sure the dataset can be converted into this format. The training, validation, and test splits are handled by modify_data.train_test_split(). An example using the Iris dataset is provided in the main file to show how DDF is executed. When switching to a different dataset, simply replace the Iris preprocessing section with the corresponding preprocessing for the new data. To keep the project organized, we recommend placing preprocessing scripts in a folder called datasetpreprocessing and importing them as needed.

#2a: The number of trees, number of epochs, and the set of features used for retraining can be configured in the main file through the variables: NoT, epochs, and Shapley_chosen_features.

#2b: Tree-specific properties—such as maximum depth, minimum gain, and minimum samples per split are defined in the DecisionTree class inside trees.py (maxdepth, min_gain, min_samples).

#3: DDF currently supports two heuristics for the retraining strategy: averageNode_Depth and Symmetrical Uncertainty. Only one heuristic is used per run, and it can be selected in the main file using the boolean Use_AD, where True enables averageNode_Depth.

#4: Once the above steps are completed, DDF can be run directly by executing DDF_main.py.



