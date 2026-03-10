*Machine Learning and AI Projects*

This repository contains assignments and projects from my AI510 course focused on machine learning, MLOps, and ML pipelines.

Topics covered:
- Machine Learning models
- CI/CD for ML
- Experiment tracking
- Model monitoring

*PE08*

Project Description

The goal of this exercise is to improve the previous HOS08 pipeline by integrating a GitHub Actions workflow to simulate a basic MLOps CI/CD process.

The application uses the Iris dataset and a machine learning model to simulate training, deployment, and monitoring of a model in a cloud-like environment. The workflow automatically runs the pipeline whenever code is pushed to the repository.

Objective

The main objective of this assignment is to demonstrate basic concepts of:

- Continuous Integration and Continuous Delivery (CI/CD)
- Automated ML pipeline execution using GitHub Actions
- Model training and deployment simulation
- Monitoring simulation using generated metrics

 Output

The scripts print execution details and simulate logs for:

- Model training
- Model deployment
- Monitoring metrics (latency and accuracy)

Example output from running the pipeline locally:
Cloud Monitoring: Latency: 31.22 ms | Accuracy: 0.87


*PE06*

Project Description

The goal of this exercise is to improve the previous pipeline HOS06 by adding model versioning and metadata logging to simulate a basic MLOps workflow.

The application uses the Iris dataset and a machine learning model to simulate training, versioning, deployment, and monitoring of a model in an AWS-like environment.

Objective

The main objective of this assignment is to demonstrate basic concepts of:

* Model versioning using timestamped folders
* Traceability of model training runs
* Metadata logging (model name, timestamp, accuracy)
* Simulated MLOps pipeline (S3, SageMaker, CloudWatch)
* Observability through logs and metrics

The project simulates a simple MLOps pipeline:

1. Train model and evaluate accuracy
2. Save model locally
3. Simulate upload to S3
4. Log metadata to model registry
5. Simulate deployment with SageMaker
6. Simulate monitoring with CloudWatch

Output

The script prints execution details and writes metadata to:

model_registry.log

Example log entry:

20260222_062139, model=iris_model.pkl, accuracy=1.00


*PE05 Project Description*

This project was developed for the PE05 Programming Exercise.

The goal of this exercise is to improve the monitoring system from the previous assignment "HOS05" by adding input validation, error logging, and proper HTTP responses to simulate how production MLOps systems detect and track invalid API requests.

The application uses the Iris dataset and a Random Forest classifier to simulate a monitored machine learning prediction service.

Invalid requests are detected, logged, and handled gracefully.

*Objective*

The main objective of this assignment is to demonstrate basic concepts of:

- API input validation
- Error handling in production systems
- Logging using logging.error()
- Monitoring of model requests
- Observability in machine learning services

*New Feature*

A validation mechanism was added to detect malformed or missing JSON inputs in the `/predict` endpoint.

When invalid input is detected:

- The error is logged using logging.error()
- The API returns an HTTP 400 response
- The issue is recorded in the log file for monitoring

*Logging and Monitoring*

The application records:

- Valid prediction requests
- Invalid input errors
- Request timestamps
- Latency information

Logs are stored in:

logs/app.log

*Output*

The API provides:

- Prediction results for valid requests
- JSON error responses for invalid requests
- Monitoring metrics through `/monitor`
- Health status through `/health`



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
