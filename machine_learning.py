import numpy
import random

import matplotlib.pyplot as plt


def studentReg(ages_train, net_worths_train):
  from sklearn.linear_model import LinearRegression
  reg = LinearRegression()
  reg.fit(ages_train, net_worths_train)
  return reg
random.seed(42)
numpy.random.seed(42)

ages = []
for ii in range(100):
  ages.append( random.randint(20,65) )

net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]

ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

reg = studentReg(ages_train, net_worths_train)

print("Coefficient",reg.coef_)
print("Slope",reg.intercept_)


print("Slope")

print("Testig data",reg.score(ages_test, net_worths_test))
print("Training data",reg.score(ages_train, net_worths_train))

plt.plot(ages_test,reg.predict(ages_test))
plt.xlabel("Ages")
plt.ylabel("Net Worth")
plt.show()