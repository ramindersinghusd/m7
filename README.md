## Predicting Wine Quality Using Machine Learning: An Exploratory Study and Model Comparison
This project is a part of the AAI-500-IN2 course in the Applied Artificial Intelligence Program at the University of San Diego (USD). 

**Project Status: [Completed]**

### Installation
To Clone project from GitHub:
1. Github url for code repo: https://github.com/ramindersinghusd/m7
2. To run locally, clone the repo as below
```
git init
git clone https://github.com/ramindersinghusd/m7.git
```

Below local env setup used for development.

1. Install python: 3.11.3
2. Installed VSCode
3. Add Python and jupyter extension to VSCode
4. Set kernel in VSCode to execute notebook
5. conda install -n base ipykernel jupyter
6. conda -V >> conda 23.5.2
7. pip install jupyter notebook pandas numpy matplotlib scipy scikit-learn pandoc
nbconvert[webpdf] nbconvert notebook-as-pdf seaborn xgboost shap openpyxl
8. goto project folder and run >> jupyter notebook
9. This will open default browser at http://localhost:8080/tree 
10. Select the required notebook file and double click to run
11. This will open in new tab, the run all cell or execute specific cell

### Project Objective

The main purpose of this project is Predicting Wine Quality Using Machine Learning via An Exploratory Study and Model Comparison.

To analyze the dataset of white wines (that has several chemical measurements, such as acidity, sugar, pH, and alcohol) for wine features-quality and quality-binary-scores, 3 models (Linear Regression, Random Forest Regressor, and XGBoost Regressor) were used. Regression and Classification performance were checked which results in Random Forest Regressor outperformed others.

The results show that machine learning can effectively predict wine quality by using data. It serves winemakers by helping them manage quality and consumers by ensuring they know what they are purchasing. 

### Team members
- George David Asirvatharaj [gdavidasirvatharaj@sandiego.edu]
- Hemlatha Kaur Saran [hsaran@sandiego.edu]
- Raminder Singh [ramindersingh@sandiego.edu]

### Methods Used
Markdown is a lightweight markup language based on the formatting conventions
that people naturally use in email.
As [John Gruber] writes on the [Markdown site][df1]


### Technologies

- [Python] - framework to use statistics lib and visualizations
- [VisualStudio Code] - OpenSource text editor
- [Jupyter Extension] - to build, manage and run notebooks
- [python libs] - jupyter notebook pandas numpy matplotlib scipy scikit-learn pandoc
nbconvert[webpdf] nbconvert notebook-as-pdf seaborn xgboost shap openpyxl
- [Chrome Browser] - to run the notebook and to convert to pdf
- [Github.com] - to maintain the code repo and to upload the final submissions
- [Zoom] - for collaborations and recording

### Project Description

**_Extracts taken from project report_**

The main issue being investigated is the choice between predicting the exact quality score using regression or using classification to tell wines apart based on quality. Scores of 7 or higher designate good quality in this report. We aim to analyze the dataset, create different machine-learning models for both regression and classification, compare them, and measure their results (Bhardwaj et al., 2022). Machine learning methods, and more specifically, Random Forest models, can predict the quality of white wine using chemical information, which is helpful for both growers and wine drinkers.

In the first phase of analysis, we checked the dataset to see if any information was missing or incorrect. There are 12 columns in the white wine dataset: eleven feature columns with wine property information and just one target column for wine quality. Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol content are part of the feature columns (Yavas et al., 2025). The quality variable, considered the target, is given as a number between 0 and 10.

The data used contains 4,898 white wine samples that have a variety of chemical features. Most fixed acidity measurements are around 6.85, but they may be as low as 3.8 or as high as 14.2. Volatile acidity is usually 0.28, and most wines have alcohol levels of 8.4% to 14.2%, averaging 10.5%. Most wines are in the middle of the quality range, based on the fact that quality scores run from 3 to 9, with an average score of around 5.88. The fact that sulphates and pH do not always stay the same indicates they may affect the wine's flavor.
```
Summary statistics:
       fixed acidity  volatile acidity  citric acid  residual sugar  \
count    4898.000000       4898.000000  4898.000000     4898.000000   
mean        6.854788          0.278241     0.334192        6.391415   
std         0.843868          0.100795     0.121020        5.072058   
min         3.800000          0.080000     0.000000        0.600000   
25%         6.300000          0.210000     0.270000        1.700000   
50%         6.800000          0.260000     0.320000        5.200000   
75%         7.300000          0.320000     0.390000        9.900000   
max        14.200000          1.100000     1.660000       65.800000   

         chlorides  free sulfur dioxide  total sulfur dioxide      density  \
count  4898.000000          4898.000000           4898.000000  4898.000000   
mean      0.045772            35.308085            138.360657     0.994027   
std       0.021848            17.007137             42.498065     0.002991   
min       0.009000             2.000000              9.000000     0.987110   
25%       0.036000            23.000000            108.000000     0.991723   
50%       0.043000            34.000000            134.000000     0.993740   
75%       0.050000            46.000000            167.000000     0.996100   
max       0.346000           289.000000            440.000000     1.038980   

                pH    sulphates      alcohol      quality  
count  4898.000000  4898.000000  4898.000000  4898.000000  
mean      3.188267     0.489847    10.514267     5.877909  
std       0.151001     0.114126     1.230621     0.885639  
min       2.720000     0.220000     8.000000     3.000000  
25%       3.090000     0.410000     9.500000     5.000000  
50%       3.180000     0.470000    10.400000     6.000000  
75%       3.280000     0.550000    11.400000     6.000000  
max       3.820000     1.080000    14.200000     9.000000  
```
Distribution plots highlight clear patterns in the most important physicochemical properties of white wine. For fixed acidity, volatile acidity, and citric acid, there are more cases with lower values and fewer with higher values, showing a distribution that is taller on the left than on the right.

The correlation matrix demonstrates the relationship between different wine properties and quality. A strong and positive relationship (0.44) exists between higher alcohol content and increased perceived quality

The boxplots reveal that multiple chemical measurements are linked to the quality of the wine. The best wines usually contain higher amounts of alcohol than lesser qualities

The project involved investigating and solving problems of the types of regression and classification. In regression, we tried to estimate the wine quality, while in classification, we marked wines as Good or Bad if their score was higher or lower than 7. The regression approach and the tested models for predicting quality values are at the center of this section.
Three models were tested: Linear Regression, Random Forest Regressor, and XGBoost Regressor. The fact that Linear Regression is simple and easy to explain is why it is considered the first model. However, it considers that features and the target are linked linearly, though this may not be sufficient for wine quality data. Because the Random Forest Regressor is an ensemble of trees, it handles nonlinear problems and multiple interactions and tends to improve accuracy. Another important type of ensemble method, XGBoost, applies gradient boosting and usually performs well on structured information due to its strong optimization and regularization.

Evaluation of the models took into account three regression indicators: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²). They measure how accurate the predictions are and the number of changes in the dataset that the model can explain. 

The classification task focused on comparing "Good" wines (quality ≥ 7) with "Bad" wines using logistic regression, Random Forest, and XGBoost classifiers. The highest accuracy (0.893), precision (0.859), recall (0.643), and F1-score (0.736) were achieved by Random Forest. 

The confusion matrix for the Random Forest classifier shows that out of the total samples, 729 wines were correctly classified as "Bad" (true negatives), and 146 wines were correctly identified as "Good" (true positives). However, 24 "Bad" wines were misclassified as "Good" (false positives), and 81 "Good" wines were misclassified as "Bad" (false negatives). 

Feature importance plot illustrates how each physicochemical property affects the prediction of wine quality. Alcohol has a much bigger impact than any other factor in this analysis. This agrees with research that shows a good connection between alcohol concentration and the taste of a wine. 

Main challenges were (1) having a single way of running jupyter notebook, and  (2) adopting the github use of the work as few team members were very new to git and github. This also include to understand different ways to create organizations, provide access inside github which took more time than expected.

For full details refer to [Final-Project-Report-Team-4.pdf]

### License
MIT
- Top-level fiter expression: _owner:ramindersinghusd license:MIT_

### Acknowledgments
We as Team-4 members really thankful to Prof Azka A for her support, guidance and  making it some easily for us to understand the Probability and Statistics fundamentals throughout this module - AAI-IN2-500 

