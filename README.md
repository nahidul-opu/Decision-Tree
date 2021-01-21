# Decision-Tree
Pure Python implementation of Decision Tree  algorithm. \
accepts pandas DataFrame. \
API is similar to scikit-learn. \
### Example

from DecisionTree import DecisionTree \
dtf=DecisionTree() \
dtf.fit(x_train,y_train) \
y_pred = dtf.predict(x_train.probability=False) \
df.print() #print the tree \
