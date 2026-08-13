# Machine Learning Pipeline - Iris Classification

## Overview
This project demonstrates a complete machine learning pipeline using the Iris dataset and a Decision Tree Classifier. It covers the essential steps of building and evaluating a machine learning model.

## Workflow
- Load the Iris dataset
- Split the dataset into training and testing sets
- Select a baseline machine learning algorithm
- Train the model
- Generate predictions on test data
- Evaluate prediction quality

## Technologies Used
- Python
- Scikit-learn

## Dataset
The project uses the built-in **Iris dataset** available in `scikit-learn`.

## Model
**Decision Tree Classifier**

## Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## Project Structure
```text
.
├── day11.py
└── README.md
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhi-repo-cs/ABTalks-60DaysChallenge/day11.git
   ```

2. Navigate to the project folder:
   ```bash
   cd day11
   ```

3. Install dependencies:
   ```bash
   pip install scikit-learn
   ```

4. Run the script:
   ```bash
   python day11.py
   ```

## Expected Output
The model will display:
- Accuracy score
- Confusion matrix
- Classification report

Example:

```text
Accuracy: 1.0

Confusion Matrix:
[[10 0 0]
 [ 0 9 0]
 [ 0 0 11]]
```

## Learning Outcome
This project demonstrates the complete end-to-end machine learning workflow, from data preparation to model evaluation, and serves as a beginner-friendly introduction to supervised learning.

## License
This project is for educational purposes.