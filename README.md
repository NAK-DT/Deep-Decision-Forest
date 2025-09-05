# Deep-Decision-Forest
Deep-Decision-Forest (DDF) is a custom ensemble tree model that takes inspiration from Deep Learning and Deep Forest (Zhou & Feng). Unlike traditional decision trees or random forests, DDF is designed to explore hierarchical feature learning through multiple layers of decision trees.

The model introduces layered training and retraining strategies, allowing decision trees to progressively refine their learned representations, similar to how deep neural networks refine features across layers. Instead of gradient-based backpropagation, DDF leverages tree retraining guided by feature importance. 


DDF has been evaluated against classical models (Decision Tree, Random Forest) and Deep Forest variants.
The table below shows average accuracy ± standard deviation across several datasets:
<img width="732" height="226" alt="Image" src="https://github.com/user-attachments/assets/ad66b71b-f4de-42b7-a2b5-036d1fce21c1" />
