import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf= svm.SVC(gamma=0.0001,C=100)
x,y = digits.data[:-10],digits.target[:-10]
clf.fit(x,y)
print('prediction:',clf.predict(digits.data[-1]))
plt.imshow(digits.images[-1])
plt.show()
