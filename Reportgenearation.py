# demonstration of calculating metrics for a neural network model using sklearn
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
    n_test = 500
    trainX, testX = X[:n_test, :], X[n_test:, :]
    trainy, testy = y[:n_test], y[n_test:]
    return trainX, trainy, testX, testy

def get_model(trainX, trainy):
    model = Sequential()
    model.add(Dense(100, input_dim=2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainy, epochs=300, verbose=0)
    return model

trainX, trainy, testX, testy = get_data()
model = get_model(trainX, trainy)

yhat_probs = model.predict(testX, verbose=0).flatten()
yhat_classes = (yhat_probs > 0.5).astype("int32")

accuracy = accuracy_score(testy, yhat_classes)
print('Accuracy: %f' % accuracy)
precision = precision_score(testy, yhat_classes)
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes)
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes)
print('F1 score: %f' % f1)
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)
auc = roc_auc_score(testy, yhat_probs)
print('ROC AUC: %f' % auc)

f, ax = plt.subplots(figsize=(8, 8))
matrix = confusion_matrix(testy, yhat_classes)
sns.heatmap(matrix, annot=True, linewidths=0.01, cmap="Blues", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
print(matrix)
