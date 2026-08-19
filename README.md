

Iris Classification



A beginner ML project that explores the classic Iris dataset and builds a Logistic Regression model to classify flower species based on measurements.



Overview



This project covers the full basic ML workflow:



Exploratory Data Analysis (EDA) with pairplots and a correlation heatmap

Train/test split

Logistic Regression model training

Evaluation via accuracy score and classification report

5-fold cross-validation to check model stability

Dataset



The Iris dataset contains 150 samples across 3 species (setosa, versicolor, virginica), with 4 features per sample:



Sepal length (cm)

Sepal width (cm)

Petal length (cm)

Petal width (cm)



Loaded directly via sklearn.datasets.load\_iris().



Project Structure

iris-classification/

├── iris\_classifier.py   # Main script: EDA, training, evaluation

├── README.md

└── .gitignore

Requirements

numpy

pandas

matplotlib

seaborn

scikit-learn



Install with:



bash

pip install numpy pandas matplotlib seaborn scikit-learn

What the script does

Load data — loads Iris into a pandas DataFrame with feature names and species labels.

EDA — generates a pairplot (colored by species) and a correlation heatmap of the four features.

Train/test split — 80/20 split, random\_state=42 for reproducibility.

Model training — fits a LogisticRegression classifier on the training set.

Evaluation — reports accuracy and a full precision/recall/F1 breakdown per species.

Cross-validation — runs 5-fold cross-validation to check the model performs consistently across different data splits, not just one lucky split.

Results



Logistic Regression achieves very high accuracy on this dataset (Iris is a well-separated, well-known "easy" classification benchmark), with strong precision/recall across all three species.



Branches

main — stable, working pipeline

Cross-validation — adds cross-validation scoring on top of the Logistic Regression model

Next Steps

Try alternative models (Random Forest, SVM, KNN) for comparison

Add a confusion matrix visualization

