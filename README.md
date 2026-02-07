*PE04 Project Description*

This project was developed for the PE04 Programming Exercise.

The goal of this exercise is to improve the Kaizen cycle from the previous assignment HOA04 by adding traceability and observability to the machine learning model update process.

The application uses the Iris dataset and a Random Forest classifier to simulate continuous improvement of a machine learning model.

Each execution compares a new model with the current best model. The model is only replaced if the new accuracy is higher.

*Objective*

The main objective of this assignment is to demonstrate basic concepts of:
- Continuous improvement (Kaizen) in machine learning
- Model performance comparison
- Traceability using timestamps
- Observability through performance logging

*New Feature*

A logging mechanism was added to record when each model evaluation occurred.

Each run stores:
- Old model accuracy
- New model accuracy
- Whether the model was improved
- Timestamp of the evaluation

*Output*

The script prints the comparison results and saves the log to:
model/performance_log.csv


*PE03 Project Description*

This project was developed for the PE03 Programming Exercise.  
The goal of this exercise was to improve the existing machine learning deployment by adding an automated validation step to the GitHub Actions CI/CD workflow.

The application trains a machine learning model using the Iris dataset and serves predictions through a Flask API. In this exercise, the CI/CD pipeline was extended to automatically verify that the deployed model produces correct predictions after each build.

*Objective*

The main objective of this assignment is to demonstrate basic concepts of:

Continuous Integration and Continuous Delivery (CI/CD)
GitHub Actions workflows
Automated testing of machine learning models
Integration testing using cURL
Model validation after deployment

*New Feature*

A new validation step was added to the GitHub Actions workflow (`cd_pipeline.yml`).

After the model is trained and the Flask API is launched, the workflow sends a test request to the `/predict` endpoint using cURL and verifies the returned prediction.

If the prediction is correct, the workflow completes successfully.  
If the prediction is not as expected, the workflow fails automatically.

This simulates a Continuous Delivery check that ensures every new commit builds, trains, and validates the deployed model.


*Output*

A successful GitHub Actions run includes a validation step labeled *Validate Model Output*, confirming that the model prediction matches the expected result.



*PE02*
Project Description

This project was developed for the PE02 Programming Exercise. The goal of this project was to improve the Iris machine learning API created in PE01 by adding a new route that provides runtime information from the container.

The application runs a trained machine learning model using the Iris dataset and exposes a Flask API. In this exercise, a new /runtime endpoint was added to return information about the execution environment.

*Objective*

The main objective of this assignment is to demonstrate basic concepts of:

Extending an existing Flask API

Working with Docker containers

Accessing runtime and system information

Creating new REST endpoints

*New Feature*

A new route /runtime was added.
This route returns a JSON response containing:

Python version

Operating system platform

Container hostname

Installed package versions (Flask, scikit-learn, Joblib)

*Output*

The API returns JSON responses.

The /predict route returns:

A numeric prediction label

A human-readable species name

The /runtime route returns:

Runtime and environment information from the container



*PE01*

Project Description

This project was developed for the PE01 Programming Exercise. The goal was to build a simple machine learning application using the Iris dataset and deploy it as a Flask API.

The application trains a classification model and exposes an API endpoint that receives flower measurements and returns both the predicted class label and the corresponding species name.

Objective

The main objective of this assignment is to demonstrate basic concepts of:

Machine learning model training

Model persistence

API development with Flask

JSON-based prediction responses

Output

The API returns a JSON response containing:

A numeric prediction label

A human-readable species name
