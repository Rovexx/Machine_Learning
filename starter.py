"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Decision tree en logistic regression
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)

STUDENTNUMMER = "0940512"

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

#print(X)

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]
y = X[...,1]

# Hoeveel clusters
kmeans = KMeans(n_clusters=6)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_
colors = ["r.", "g.", "b.", "c.", "m.", "y.", "k."]

# teken de punten
for i in range(len(x)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = "10")

plt.axis([min(x), max(x), min(y), max(y)])
plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", s = 150, linewidths = 25, zorder = 10)

plt.show()


# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)
# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)


# leer de classificaties
x_2 = X[...,0]
y_2 = X[...,1]

for i in range(len(x_2)):
    plt.plot(x_2[i], y_2[i], 'k.')

plt.axis([min(x_2), max(x_2), min(y_2), max(y_2)])

# toon alle punten zonder classificatie
plt.show()

# DECISION TREE
# train deze punten in een decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

# voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#   echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#   bijvoordeeld Y_predict
Y_predict = clf.predict(X)
clf_score = accuracy_score(Y, Y_predict)
print("Decision tree accuratie (score): " + str(clf_score))

# vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
plt.scatter(x_2, y_2, c = Y_predict, s = 10)
plt.show()

# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
# je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
# geen echte Y-waarden gekregen hebt.
# onderstaande code stuurt je voorspelling naar de server, die je dan
# vertelt hoeveel er goed waren.
Y_predict = clf.predict(X_test)
Z = Y_predict
# Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (decisionTree): " + str(classification_test))
