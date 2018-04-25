# Animal DataSet Classification

In this assignment, classification algorithms such as logistic regression, decision tree, random forest and naïve Bayes classifier are modeled on the dataset using scikitlearn package of python. The results of the classifiers have been evaluated using cross-validation technique and statistical t-test.

DataSet:
--------

Used the Animals classification [dataset](https://www.fs.fed.us/pnw/starkey/). The target variable is the last column in the
CSV file. Each row in the file labeled with one of the following categories:
1. Elk (52%)
2. Deer (28%)
3. Cattle (20%)
<br />
Other information about the dataset:<br />
* Number of instances: 5135
* Number of features: 25
* Number of classes: 3

Useful Python packages:
* **Numpy:** multidimensional arrays, vector and matrix operations
* **Pandas:** data manipulation and analysis
* **Scikit-learn:** machine learning library for classification, regression, clustering, feature selection and much more.
* **Matplotlib and Seaborn:** For plotting the graphs.

Task Performed:
---------------
(a) Splitted the data randomly into a training set and a testing set (e.g. 70%-30%). Train all
classifiers (Logistic Regression, Naïve Bayes, Decision Tree and Random Forests)
using the default parameters using the train data. Reported the confusion matrix and
accuracy for both train and test data. Compared the train and test accuracy. to identify the
difference between train and test accuracy. <br />
(b) Used 10-fold cross-validation, train and evaluate all classifiers. Compare the
accuracy of the methods in terms of mean (𝜇) and standard deviation (𝜎) of
accuracy in 10 folds. Eventually used a statistical significance test (e.g. student’s ttest)
and determine whether the methods are significantly different or not. Use 𝛼 =
0.05 as the significance threshold. Furthermore have applied the significance test, on the
classifier which had the best average performance, and compared it to all the remaining
classifiers.<br />
(c) Traineedd a Random Forest using a 10-fold cross-validation with the 10, 20, 50 and 100
trees (e.g. number of estimators in the scikit package) and reported the mean
accuracies.

Documentation of the code is [here](https://github.com/dalalbhargav07/Dalhousie---Machine-Learning-with-Big-Data/blob/master/Assignment%201%20-%20Animal%20Dataset%20Classification/A1_BhargavDalal_B0785773.pdf).
