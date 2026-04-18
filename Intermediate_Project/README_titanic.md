# Titanic Survival Prediction

## Description
Predict passenger survival on the Titanic using demographic and ticketing features. The project covers end-to-end data science: cleaning, feature engineering, EDA, multi-model training, and evaluation.

## Dataset
- **Source:** [Kaggle / datasciencedojo GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) (fallback: seaborn built-in)
- **Description:** 891 passengers with features including age, sex, passenger class, fare, embarkation port, and family size. Target variable is `Survived` (0 = Died, 1 = Survived).

## Steps Performed
1. **Data Cleaning** — Age imputed using median grouped by `Pclass` & `Sex`; `Embarked` filled with mode; `Cabin`, `Name`, `Ticket`, `PassengerId` dropped
2. **Feature Engineering** — `FamilySize` (SibSp + Parch + 1), `IsAlone` flag, `AgeBand` and `FareBand` bins, label encoding of categorical columns
3. **Exploratory Data Analysis** — 9-panel visualization: survival counts, survival by sex/class/family size/embarkation, age & fare distributions, Pclass×Sex heatmap, correlation heatmap
4. **Model Building** — Logistic Regression, Random Forest (200 trees), Gradient Boosting (200 estimators); 80/20 stratified split; StandardScaler; 5-fold cross-validation to select best model

## Results
- **Overall survival rate:** ~38%
- **Female survival rate:** ~74% vs. Male ~19% — sex is the strongest predictor
- **1st class survival:** ~63% vs. 3rd class ~24%
- **Travelling with family** outperforms travelling alone
- **Best model:** Gradient Boosting (typically ~83% test accuracy)
- **Metrics:** Accuracy, Precision, Recall, F1-score (classification report)

## Tools Used
- Python
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, StandardScaler, cross_val_score)

## Conclusion
Sex, passenger class, and fare are the dominant survival predictors, consistent with the historical "women and children first" evacuation policy. Gradient Boosting delivers the best generalization, and feature importance confirms `Sex`, `Fare`, and `Pclass` as top contributors.

## Author
Prasan Kanaparthi
