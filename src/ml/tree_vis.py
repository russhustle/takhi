# Reference: https://scikit-learn.org/stable/modules/tree.html

from sklearn.datasets import load_iris
from sklearn import tree
import graphviz

# Train
iris = load_iris()
X, y = iris.data, iris.target
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Visualize the decision tree
dot_data = tree.export_graphviz(
    decision_tree=clf,
    out_file=None, 
    feature_names=iris.feature_names,  
    class_names=iris.target_names,  
    filled=True,
    rounded=True,  
    special_characters=True,
)  
graph = graphviz.Source(dot_data)  
graph
