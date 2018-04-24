# GeoSpatial-Data-Classification

In this project, have applied the concept of feature engineering for pre-processing and cleaning of the data. After pre-processing of the data, it was hierarchically classified into a tree structure by exploring the data to find the relation among the classes. Further applied classification algorithm such as Random Forest, Decision Tree on the hierarchical tree structure classified data as well as flat structured data. Compare model for tree structure as well as flat structure using statistical t-test.

DataSet:
--------
Use the provided Geolife dataset. The target variable is the last column in the CSV file. Each row in the file labeled with one of the following categories:
1. bus (29%)
2. car (11%)
3. walk (36%)
4. taxi (5%)
5. subway (7%)
6. train (12%)
<br />
Other information about the dataset:
* Number of trajectory points: 4.485.796
* Number of classes: 6

Installation notes and Packages Required:
-----------------------------------------
* **Numpy:** multidimensional arrays, vector and matrix operations
* **Pandas:** data manipulation and analysis
* **Scikit-learn:** machine learning library for classification, regression, clustering, feature selection and much more.
* **Matplotlib & Seaborn:** For plotting the graphs.

<br />
Required to unzip the geolife_raw data file to run the program successfully.

Task Peformed:
--------------

1. Feature Engineering:
  * Grouped the trajectories by user id and day and compute the following point features: (i) distance traveled (e.g. haversine, in meters); (ii) speed (m/s); (iii) acceleration(m/s2); (iv) bearing (0 to 360 degrees).
  * Created sub-trajectories by class using the daily trajectories and compute the trajectory features as follows:
      a. Discarded sub-trajectories with less than 10 trajectory points.
      b. For each point feature, computed the minimum, maximum, mean, median and standard deviation. Those 20 values (5 statistical measures x 4 point features) which are later used for classification.
  * Explored the data and compare the trajectory features values by class to detect similarities or significant differences between the classes. It was explored using various graphs.
  
2. Hierarchical classification:
  * After evaluating the trajectory features, proposed a hierarchy to classify the data which was choosed according to the graphs. Detail explanation of each class at each level of the hierarchical treee structure is given in the document.
  * Implemented the tree structure and then compared the tree structure and flat structure using Random Forest and Decision Tree classifier.
  * Performed a multiclass evaluation and a significance test (e.g. paired t-test) for each classifier. Use a ten-fold cross-validation with stratification.

       
#### Documentation of the project is [here] (https://github.com/dalalbhargav07/Dalhousie---Machine-Learning-with-Big-Data/blob/master/Project%20-%20GeoSpatial%20Data%20Classification/Project%20Report_Bhargav%20Dalal_B00785773.pdf).


