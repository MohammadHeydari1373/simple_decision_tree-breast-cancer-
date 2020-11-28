#mohammad reza heydari
#mr.heidari1373@gmail.com
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


br = load_breast_cancer()
x = br.data
y = br.target
n_classes = br.target.size

test_size = 1 / 3
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                    shuffle=True, stratify=y,
                                                    random_state=0)



# Fit the classifier
clf = DecisionTreeClassifier(random_state=300)
model = clf.fit(X_train,y_train)

text_representation = tree.export_text(clf)
print(text_representation)



import graphviz

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=br.feature_names,  
                                class_names=br.target_names,
                                filled=True)


graph = graphviz.Source(dot_data, format="png") 
graph

print(graph.render("decision_tree_graphivz"))

from google.colab import drive
drive.mount('/content/drive')



from dtreeviz.trees import dtreeviz 

viz = dtreeviz(clf, x, y,
                target_name="target",
                feature_names=br.feature_names,
                class_names=list(br.target_names))

viz

viz.save("decision_tree.svg")



y_pred = clf.predict(X_test)



#precision,recall ,f1-score , Acuraccy


from sklearn.metrics import accuracy_score, classification_report
report_test = classification_report(y_test, y_pred)
print('Testing\n%s' % report_test)
accuracy_test = accuracy_score(y_test, y_pred)
print(f"accuracy for testing data : {accuracy_test}")