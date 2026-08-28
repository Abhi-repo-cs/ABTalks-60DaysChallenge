## Day 26 – Hyperparameter Tuning

This project demonstrates the optimization of a Random Forest classification model using `RandomizedSearchCV`.

A baseline Random Forest model was first trained using default hyperparameters. Its performance was then compared with a tuned model obtained by searching across multiple hyperparameter combinations using 5-fold cross-validation.

The tuning process explored parameters such as the number of estimators, maximum tree depth, minimum samples required for splitting, minimum samples per leaf, feature selection strategy, and bootstrap sampling.

The tuned model was evaluated on the same held-out test set as the baseline model using Accuracy, Precision, Recall, and F1 Score.

### Key Learning

Hyperparameter tuning provides a systematic way to improve machine learning models instead of relying on default configurations. However, optimization introduces additional computational cost and does not always produce a large improvement in test performance.

The main goal is therefore to find a practical balance between predictive performance, computational cost, model complexity, and reliability.

### Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Random Forest
* RandomizedSearchCV

### Conclusion

The experiment shows how systematic hyperparameter optimization can improve or validate the performance of a machine learning model. Comparing the tuned model against a baseline provides evidence of whether the additional computational cost of optimization is justified.
