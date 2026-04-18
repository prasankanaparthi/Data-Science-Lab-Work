# California Housing Price Prediction

## Description
Predict median house values across California districts using the 1990 Census dataset. The project spans four phases: preprocessing, model training, evaluation, and SHAP-based interpretability.

## Dataset
- **Source:** Kaggle (`housing.csv`) / scikit-learn `fetch_california_housing`
- **Description:** 20,640 rows × 10 columns representing California census districts. Target variable is `median_house_value` (USD). Only `total_bedrooms` has missing values (~1%).

## Steps Performed
1. **Data Cleaning** — Median imputation for `total_bedrooms`; binary flag for capped `housing_median_age`
2. **Feature Engineering** — Log-transforms on skewed count columns; derived ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`); one-hot encoding of `ocean_proximity`
3. **Exploratory Data Analysis** — Target distribution (raw vs. log), correlation analysis, skewness assessment (carried from Phase 0 EDA)
4. **Model Building** — 5 models × 2 target variants (raw + log) = 10 experiments: Linear Regression, Ridge, Random Forest, XGBoost, LightGBM; 80/20 stratified train-test split; StandardScaler applied
5. **Evaluation** — RMSE, MAE, MAPE, R², Adjusted R², residual plots, learning curves, MAPE by price band
6. **Interpretability** — SHAP TreeExplainer: global importance bar chart, beeswarm plot, dependence plot (`median_income`), waterfall plots

## Results
- **Best model:** XGBoost or LightGBM (log-target variant, determined at runtime)
- **Key predictors:** `median_income` (strongest, r = 0.69), `latitude`, `longitude`
- Errors are highest in the `<$100K` and `>$400K` price bands due to data sparsity and value capping
- Log-transforming the target reduces skewness and generally improves tree-model performance

## Tools Used
- Python
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (preprocessing, pipelines, metrics)
- XGBoost, LightGBM
- SHAP

## Conclusion
Gradient boosting models (XGBoost/LightGBM) with a log-transformed target deliver the best predictions. `median_income` is by far the most influential feature, followed by geographic coordinates. SHAP analysis confirms these findings and reveals non-linear interactions between income and location.

## Author
Prasan Kanaparthi
