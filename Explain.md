## Decision Tree Structure
A Decision Tree consists of:
- **Root Node**: The starting point of the tree, representing the entire dataset.
- **Internal Nodes**: Nodes where decisions are made based on feature values.
- **Leaf Nodes**: Terminal nodes that provide the output (prediction).

### Training Phase
1. **Data Splitting**: The tree splits the data into subsets based on the value of a chosen feature at each node. The feature and the split point are chosen to maximize a criterion (e.g., reduction in variance for regression trees or information gain for classification trees).

2. **Recursive Splitting**: This process is repeated recursively for each subset, creating a structure of nodes and branches until a stopping condition is met (e.g., maximum depth, minimum samples per leaf, or no further improvement in splitting).

### Prediction Phase
1. **Traversing the Tree**: To make a prediction, the model starts at the root node and follows the branches based on the feature values of the input data, making decisions at each internal node until it reaches a leaf node.

2. **Outcome at Leaf Node**: The prediction outcome is provided at the leaf node. For a regression tree, this is typically the mean of the target values in that leaf node.

### Example Walkthrough
Assuming a simplified Decision Tree for predicting whether the `Standalone_PFE_Diff` is significant (binary classification):

1. **Root Node**: 
   - The entire dataset is evaluated. 
   - The best feature and split point are selected based on a criterion (e.g., Gini impurity or entropy).

2. **Internal Node 1** (e.g., Feature: `Days`):
   - If `Days <= 10`, go to the left child node.
   - If `Days > 10`, go to the right child node.

3. **Internal Node 2** (Left Child of Node 1, e.g., Feature: `BuyCurrency_d0_EUR`):
   - If `BuyCurrency_d0_EUR == 1`, go to the left child node.
   - If `BuyCurrency_d0_EUR == 0`, go to the right child node.

4. **Leaf Nodes**:
   - Each path from the root to a leaf represents a sequence of decisions. The leaf node provides the predicted outcome.
   - Example:
     - Path: Root -> Days <= 10 -> BuyCurrency_d0_EUR == 1
     - Prediction at leaf: `Standalone_Significant = 1` (significant difference)

### Detailed Walkthrough Using Code
Using the provided code, here’s how the outcomes are determined:

1. **Data Preparation**:
   - Merging and calculating differences.
   - Setting thresholds and creating binary labels for significant differences.

2. **Training the Decision Tree**:
   - For `Standalone_PFE_Diff`:
     ```python
     tree_model_standalone = DecisionTreeRegressor(random_state=0, max_depth=5)
     grid_search_standalone = GridSearchCV(tree_model_standalone, param_grid, cv=10, scoring='accuracy')
     grid_search_standalone.fit(X, y_standalone)
     best_tree_model_standalone = grid_search_standalone.best_estimator_
     best_tree_model_standalone.fit(X, y_standalone)
     ```
   - The tree model learns the best splits based on the feature values to predict the `Standalone_Significant` label.

3. **Making Predictions**:
   - The trained model is used to make predictions on new data.
   - For each new data point, the model starts at the root node and follows the branches based on the feature values until reaching a leaf node.

4. **Exporting Tree Rules**:
   - The tree rules can be exported and interpreted to understand how decisions are made:
     ```python
     tree_rules_standalone = export_text(best_tree_model_standalone, feature_names=final_features_df.columns.to_list())
     ```
   - The exported rules show the conditions at each node and the resulting predictions.

### Example of Exported Tree Rules
```
|--- Days <= 10.50
|   |--- BuyCurrency_d0_EUR <= 0.50
|   |   |--- Standalone_Significant: 0
|   |--- BuyCurrency_d0_EUR >  0.50
|   |   |--- Standalone_Significant: 1
|--- Days >  10.50
|   |--- SellCurrency_d0_USD <= 0.50
|   |   |--- Standalone_Significant: 0
|   |--- SellCurrency_d0_USD >  0.50
|   |   |--- Standalone_Significant: 1
```

- This tree predicts `Standalone_Significant` based on the `Days`, `BuyCurrency_d0_EUR`, and `SellCurrency_d0_USD` features.

### Conclusion
The Decision Tree model makes predictions by recursively splitting the data based on feature values, following a path from the root to a leaf node, and providing the outcome at the leaf node. The rules can be exported to understand and interpret how the decisions are made at each step.

## Break Down of Gordon's Decision Tree
### Imports and Assumptions
```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Assume pfe_results_df and pfe_results_d1_df are already provided
```
- Importing necessary libraries: `pandas` for data manipulation, `numpy` for numerical operations, `DecisionTreeRegressor` and `export_text` from `sklearn.tree` for building and exporting decision trees, `OneHotEncoder` from `sklearn.preprocessing` for encoding categorical variables, `GridSearchCV` from `sklearn.model_selection` for hyperparameter tuning, and `matplotlib.pyplot` for plotting.
- Assuming that `pfe_results_df` and `pfe_results_d1_df` are preloaded DataFrames.

### Data Merging and Calculating Differences
```python
# Perform an inner join on TransactionID and Days
merged_df = pd.merge(pfe_results_df, pfe_results_d1_df, on=['TransactionID', 'Days'], suffixes=('_d0', '_d1'))

# Calculate the differences
merged_df['Standalone_PFE_Diff'] = merged_df['Standalone_PFE_d1'] - merged_df['Standalone_PFE_d0']
merged_df['Collateralized_PFE_Diff'] = merged_df['Collateralized_PFE_d1'] - merged_df['Collateralized_PFE_d0']
```
- Merging `pfe_results_df` and `pfe_results_d1_df` on `TransactionID` and `Days` columns.
- Calculating differences between `d1` and `d0` for both `Standalone_PFE` and `Collateralized_PFE`.

### Setting Thresholds and Creating Labels
```python
# Set a significant threshold based on 95th percentile of the differences

# Standalone PFE Difference
standalone_threshold = np.percentile(abs(merged_df['Standalone_PFE_Diff']), 95)

# Collateralized PFE Difference
collateralized_threshold = np.percentile(abs(merged_df['Collateralized_PFE_Diff']), 95)

# Create binary labels
merged_df['Standalone_Significant'] = (abs(merged_df['Standalone_PFE_Diff']) > standalone_threshold).astype(int)
merged_df['Collateralized_Significant'] = (abs(merged_df['Collateralized_PFE_Diff']) > collateralized_threshold).astype(int)
```
- Calculating the 95th percentile of the absolute differences for both `Standalone_PFE` and `Collateralized_PFE` to set significant thresholds.
- Creating binary labels (`0` or `1`) indicating whether the absolute differences exceed the thresholds.

### One-Hot Encoding Categorical Features
```python
# One-hot encode categorical features
categorical_features = ['BuyCurrency_d0', 'SellCurrency_d0']
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = onehot_encoder.fit_transform(merged_df[categorical_features])
encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_features)

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
```
- Defining categorical features to be one-hot encoded.
- Initializing `OneHotEncoder` with `sparse_output=False` to get a dense array and `drop='first'` to avoid multicollinearity.
- Fitting and transforming the categorical features.
- Creating a DataFrame from the encoded features.

### Combining Features and Defining Targets
```python
# Combine all features
final_features_df = pd.concat([merged_df[['Days']], encoded_df], axis=1)

# Define features and target for standalone PFE difference
X = final_features_df
y_standalone = merged_df['Standalone_Significant']
y_collateralized = merged_df['Collateralized_Significant']
```
- Combining numerical (`Days`) and encoded categorical features into a final feature set.
- Defining `X` as the feature set.
- Defining targets `y_standalone` and `y_collateralized` as the binary labels for standalone and collateralized PFE differences, respectively.

### Training Decision Trees
```python
# Train a Decision Tree for Standalone PFE Difference
tree_model_standalone = DecisionTreeRegressor(random_state=0, max_depth=5)
param_grid = {'random_state': np.arange(10)}
grid_search_standalone = GridSearchCV(tree_model_standalone, param_grid, cv=10, scoring='accuracy')
grid_search_standalone.fit(X, y_standalone)

# Get the best model
best_tree_model_standalone = grid_search_standalone.best_estimator_
best_tree_model_standalone.fit(X, y_standalone)
```
- Initializing `DecisionTreeRegressor` for standalone PFE differences with `random_state=0` and `max_depth=5` to control the depth of the tree.
- Defining a parameter grid for `random_state`.
- Performing a grid search with 10-fold cross-validation to find the best `random_state`.
- Fitting the best model on the data.

### Exporting Tree Rules for Standalone PFE Difference
```python
# Export the tree as text for Standalone PFE Difference
tree_rules_standalone = export_text(best_tree_model_standalone, feature_names=final_features_df.columns.to_list())

text = ''
text += 'Decision Tree Rules for Standalone PFE Difference:\n'
text += tree_rules_standalone
```
- Exporting the decision tree rules as text for standalone PFE difference using `export_text`.

### Training and Exporting Tree Rules for Collateralized PFE Difference
```python
# Train a Decision Tree for Collateralized PFE Difference
tree_model_collateralized = DecisionTreeRegressor(random_state=0, max_depth=5)
param_grid = {'random_state': np.arange(10)}
grid_search_collateralized = GridSearchCV(tree_model_collateralized, param_grid, cv=10, scoring='accuracy')
grid_search_collateralized.fit(X, y_collateralized)

# Get the best model
best_tree_model_collateralized = grid_search_collateralized.best_estimator_
best_tree_model_collateralized.fit(X, y_collateralized)

# Export the tree as text for Collateralized PFE Difference
tree_rules_collateralized = export_text(best_tree_model_collateralized, feature_names=final_features_df.columns.to_list())

text += '\nDecision Tree Rules for Collateralized PFE Difference:\n'
text += tree_rules_collateralized
```
- Following the same steps as above, training a decision tree and exporting rules for collateralized PFE difference.

### Printing Volatility and Correlation Matrix Text
```python
# print volatility and correlation matrix to string, concatenate and with headlines explaining them
volatilities_usd_text = 'Volatilities for USD pairs:\n'
volatilities_usd_text += str(volatilities_usd)
volatilities_usd_text += '\n\n'

correlation_matrix_usd_text = 'Correlation Matrix for USD pairs:\n'
correlation_matrix_usd_text += str(correlation_matrix_usd)
correlation_matrix_usd_text += '\n\n'

volatilities_usd_d1_text = 'Volatilities for USD pairs in d1:\n'
volatilities_usd_d1_text += str(volatilities_usd_d1)
volatilities_usd_d1_text += '\n\n'

correlation_matrix_usd_d1_text = 'Correlation Matrix for USD pairs in d1:\n'
correlation_matrix_usd_d1_text += str(correlation_matrix_usd_d1)
correlation_matrix_usd_d1_text += '\n\n'

# Combine all text
market_data_text = volatilities_usd_text + correlation_matrix_usd_text + volatilities_usd_d1_text + correlation_matrix_usd_d1_text
```
- Generating text descriptions for volatilities and correlation matrices for both days.

### Visualization Function and Visualizing Trees
```python
# Function to visualize a decision tree with improved readability
def visualize_tree(model, feature_names, title):
    plt.figure(figsize=(30, 15))  # Increase figure size for better readability
    tree.plot_tree(model, feature_names=feature_names, filled=True, fontsize=10)  # Adjust font size
    plt.title(title)
    plt.show()

# Visualize the Standalone PFE Difference Decision Tree
visualize_tree(best_tree_model_standalone, final_features_df.columns, "Decision Tree for Standalone PFE Difference")

# Visualize the Collateralized PFE Difference Decision Tree
visualize_tree(best_tree_model_collateralized, final_features_df.columns, "Decision Tree for Collateralized PFE Difference")
```
- Defining a function `visualize_tree` to plot decision trees with a larger figure size for readability.
- Visualizing the decision trees for standalone and collateralized PFE differences using the defined function.