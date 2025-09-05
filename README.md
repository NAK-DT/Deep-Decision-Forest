# Deep-Decision-Forest
Deep-Decision-Forest (DDF) is a custom ensemble tree model that takes inspiration from Deep Learning and Deep Forest (Zhou & Feng). Unlike traditional decision trees or random forests, DDF is designed to explore hierarchical feature learning through multiple layers of decision trees.

The model introduces layered training and retraining strategies, allowing decision trees to progressively refine their learned representations, similar to how deep neural networks refine features across layers. Instead of gradient-based backpropagation, DDF leverages tree retraining guided by feature importance. 


DDF has been evaluated against classical models (Decision Tree, Random Forest) and Deep Forest variants.
The table below shows average accuracy ± standard deviation across several datasets:
<img width="732" height="226" alt="Image" src="https://github.com/user-attachments/assets/ad66b71b-f4de-42b7-a2b5-036d1fce21c1" />

Across these datasets, DDF performs competitively with or better than standard ensembles. The advantage becomes clearer on tasks such as Raisin, Heart Disease, and Iris, where redundancy between features can cause Random Forests and Deep Forests to produce overlapping splits. By contrast, DDF’s retraining strategies help reduce subtree repetition, enabling better exploitation of hierarchical feature interactions.

Notably, while Random Forest and Deep Forest can still achieve high accuracy on strongly separable datasets (e.g., Mushroom, Banknote), DDF shows its strength where redundancy is more problematic. This suggests that DDF should be viewed as a complementary alternative to existing ensembles, especially in structured data domains where feature overlap limits classical approaches.
