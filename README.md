# ML-Model-w-SQL---Credit-Card-Fraud-Detection-Tool

Automated Financial Fraud Detection and Adaptation System
Project Overview
This project provides a robust, production-ready machine learning framework designed to automate the lifecycle of credit card fraud detection. The core objective is to reduce manual monitoring by implementing a self-correcting system that detects performance degradation and automatically retrains models to adapt to shifting fraud patterns or data schemas.

Repository Structure
models/: Directory for serialized model packages (active_model.pkl).

  predict_fraud.py: The central inference engine and feature alignment motherboard.
  
  pipeline.py: The automated retraining logic and feature selection engine.
  
  final_test.py: Validation framework for performance verification and compatibility testing.
  
  live_check.py: Real-time SQL integration and random record analysis.
  
  importer.py: Data ingestion utility for CSV-to-SQL database replacement.
  
  analysis.py: Initial model prototyping and exploratory feature research.
  
  fraud.ipynb: Exploratory data analysis and data science research lab.

Technical Component Analysis
Core Inference: predict_fraud.py
This module serves as the primary motherboard for the system's logic. It handles real-time inference by loading the latest model package and aligning incoming data with the required feature set. To ensure system stability, it includes a safety mechanism that identifies missing features in the input and populates them with neutral values (0.0) to prevent runtime crashes.

Adaptation Engine: pipeline.py
The pipeline is responsible for the automated retraining segment of the system. It performs dynamic feature discovery, filtering out non-numeric noise and metadata to identify valid predictors. If accuracy falls below the defined threshold, it applies SMOTE for class balancing and utilizes a RandomForestClassifier to rebuild the model until a recall score of at least 90% is achieved.

System Validation: final_test.py
This script implements a practical verification layer to ensure model reliability on new data batches. It performs a schema check to determine if the existing model is physically compatible with new input data. If it detects a schema change or a drop in recall below 90%, it automatically triggers the retraining pipeline to evolve the system to the new dataset.

Production Integration: live_check.py
The live check module acts as the interface between the SQL database and the inference engine. It validates existing database tables and extracts random records for analysis. By integrating with the motherboard function in predict_fraud.py, it provides a final output verdict, including a calculated probability of fraud for each record.

Data Ingestion: importer.py
This utility streamlines the testing process by facilitating rapid data replacement. It accepts any CSV file and overwrites the 'transactions' table within the SQLite database, allowing the developer to quickly pivot between different data versions (e.g., 2013 vs. 2023 datasets).

Research and Development: analysis.py & fraud.ipynb
These files represent the research phase of the project. analysis.py was used to understand initial model behavior and identify primary fraud indicators. The Jupyter Notebook served as a learning environment for mastering Pandas manipulation and fundamental data science principles before they were codified into the production scripts.

Operational Workflow
Data Ingestion: Use importer.py to update the SQL database with the latest transaction records.

Compatibility Verification: Execute final_test.py to assess model recall against the new data batch.

Automated Adaptation: If recall performance is insufficient or the data schema has changed, pipeline.py will automatically retrain and deploy a new model package.

Production Monitoring: Utilize live_check.py to perform randomized audits of the SQL records using the updated model.
