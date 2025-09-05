# Deep-Decision-Forest
Deep-Decision-Forest (DDF) is a custom ensemble tree model that takes inspiration from Deep Learning and Deep Forest (Zhou & Feng). Unlike traditional decision trees or random forests, DDF is designed to explore hierarchical feature learning through multiple layers of decision trees.

The model introduces layered training and retraining strategies, allowing decision trees to progressively refine their learned representations, similar to how deep neural networks refine features across layers. Instead of gradient-based backpropagation, DDF leverages tree retraining guided by feature importance. 


DDF has been evaluated against classical models (Decision Tree, Random Forest) and Deep Forest variants.
The table below shows average accuracy ± standard deviation across several datasets:
<img width="732" height="226" alt="Image" src="https://github.com/user-attachments/assets/ad66b71b-f4de-42b7-a2b5-036d1fce21c1" />

In many real-world datasets, redundant features are unavoidable. Different variables may encode overlapping information (e.g., correlated medical indicators, or duplicated categorical encodings). While redundancy is not inherently harmful, it often leads to inefficiencies in traditional ensemble models. This creates a risk where models appear complex but contribute little new information beyond what a smaller subset of features already provides.

To better capture this phenomenon, we define a Redundancy Score that quantifies overlap in a dataset by combining two complementary measures:

**Redundancy Score** = 0.5 × Correlation Score + 0.5 × PCA Score

Correlation Score measures pairwise dependencies between features (e.g., Pearson correlation).

PCA Score measures how much variance can be explained by a reduced set of principal components, reflecting global redundancy.

This combined metric allows us to categorize datasets into Low, Moderate, or High redundancy regimes.

<img width="554" height="278" alt="Image" src="https://github.com/user-attachments/assets/94f03988-53c6-4ea4-899d-4114b1365fb0" />

DDF addresses this challenge by incorporating retraining strategies that encourage trees across layers to learn complementary, hierarchical patterns instead of reusing the same redundant splits. By refining feature selection at each layer, DDF aims to exploit dataset structure more effectively, particularly in redundancy-heavy domains.

Across these datasets, DDF performs competitively with or better than standard ensembles. The advantage becomes clearer on tasks such as Raisin, Heart Disease, and Iris, where redundancy between features can cause Random Forests and Deep Forests to produce overlapping splits. By contrast, DDF’s retraining strategies help reduce subtree repetition, enabling better exploitation of hierarchical feature interactions.

Notably, while Random Forest and Deep Forest can still achieve high accuracy on strongly separable datasets (e.g., Mushroom, Banknote), DDF shows its strength where redundancy is more problematic. This suggests that DDF should be viewed as a complementary alternative to existing ensembles, especially in structured data domains where feature overlap limits classical approaches.
