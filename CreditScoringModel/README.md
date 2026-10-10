# Credit Scoring Model

## Project Overview

This project predicts credit risk using Machine Learning classification algorithms. It uses the German Credit dataset from OpenML.

## Dataset

* Total records: 1000
* Total features: 20
* Training records: 800
* Testing records: 200

## Machine Learning Models

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Model Performance

| Model               | Accuracy |
| ------------------- | -------: |
| Logistic Regression |    70.5% |
| Decision Tree       |    68.5% |
| Random Forest       |    72.0% |

Random Forest achieved the highest accuracy in this experiment.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Project Files

* `credit_scoring.py` — trains and evaluates the models
* `model_comparison.py` — generates the model accuracy graph
* `model_results.csv` — stores model evaluation results
* `model_comparison.png` — model accuracy comparison graph

## How to Run

1. Install Python.
2. Install the required libraries:
   `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run:
   `python credit_scoring.py`
4. Generate the graph:
   `python model_comparison.py`

## Disclaimer

This project is for educational purposes only. Its predictions should not be used alone to make real-world lending decisions.
