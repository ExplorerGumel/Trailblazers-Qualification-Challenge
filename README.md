# Trailblazers-Qualification-Challenge

This notebook contains a code + some modifications to the solution i submitted while competing on  Trailblazers Qualification challenge organised by zindi Africa.

# OVERVIEW
Female-headed households have been on the rise in South Africa in recent years. Compared to male-headed households, female-headed households tend to face greater social and economic challenges. Female-headed households, in general, are more vulnerable to lower household incomes and higher rates of poverty.
The South African census collects data on female headship and income levels of every household across the country every 10 years. However, it is important for policymakers and other actors to have accurate estimates of these statistics even in between census years. This challenge explores how machine learning can help improve monitoring key indicators at a ward level in between census years.

The objective of the challenge is to build a predictive model that accurately estimates the percentage of households per ward that are female-headed and living below a particular income threshold, by using data points that can be collected through other means without an intensive household survey like the census.

# DATASET

The dataset is split into a train and test set, The train set contains 2,822 sample while test set contain 1,013 samples. Each row/sample in the training set includes the following informations;

1.  dw_00 --> Percentage of dwellings of type: House or brick/concrete block structure on a separate stand or yard or on a farm.
2.  dw_01 --> Percentage of dwellings of type: Traditional dwelling/hut/structure made of traditional materials.
3.  dw_02 --> Percentage of dwellings of type: Flat or apartment in a block of flats.
4.  dw_03 --> Percentage of dwellings of type: Cluster house in complex.
5.  dw_04 --> Percentage of dwellings of type: Townhouse (semi-detached house in a complex)
6.  dw_05 --> Percentage of dwellings of type: Semi-detached house.
7.  dw_06 --> Percentage of dwellings of type: House/flat/room in backyard.
8.  dw_07 --> Percentage of dwellings of type: Informal dwelling (shack in backyard).
9.  dw_08 --> Percentage of dwellings of type: Informal dwelling (shack not in backyard  e.g. in an informal/squatter settlement or on a farm)
10. dw_09 --> Percentage of dwellings of type: Room/flatlet on a property or larger dwelling/servants quarters/granny flat.
11. dw_10 --> Percentage of dwellings of type: Caravan/tent.
12. dw_11 --> Percentage of dwellings of type: Other.
13. dw_12 --> Percentage of dwellings of type: Unspecified.
14. dw_13 --> Percentage of dwellings of type: Not applicable.
15. psa_00 --> Percentage listing present school attendance as:  Yes
16. psa_01 --> Percentage listing present school attendance as:  No
17. psa_02 --> Percentage listing present school attendance as:  Do not know
18. psa_03 --> Percentage listing present school attendance as:  Unspecified
19. psa_04 --> Percentage listing present school attendance as:  Not applicable
20. stv_00 --> Percentage of households with Satellite TV:  Yes
21. stv_01 --> Percentage of households with Satellite TV:  No
    >.
    >.
    >.
    >.
61.target --> Percentage of women head households with income under R19.6k out of total number of households

# APPROACH AND RESULTS

Once suitable preprocessing and feature extraction techniques were applied, three distinct methodologies were utilized in the following manner.
1. An AdaBoost classifier with logistic Regression as a based estimator which achieved mean squared error of 17.3.

2. A simple Multi-layer perceptrons runs for 1k epochs with early stopping and learning rate scheduler callbacks which achieved a mean squared error of 12.3

3. And a convolution layers model which achieved a mean squared error of 8.6

# CONCLUSION

Participating in this challenge was a great learning experience for me. I gained valuable hands-on experience in building a machine learning model and working with real-world datasets. I am grateful for the opportunity and look forward to participating in more challenges in the future.

Feel free to check out my code in this repository and let me know your thoughts!
