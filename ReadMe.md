### CHURN MODEL ###
The aim of this repo is to demonstrate how to build a churn model, when there is only transactional data available i.e. TransactionID, CustomerID,Transaction Date, Transaction Amount. This scenario is in fact very common in practice, and although what we have here is built for a Churn Model, the features could be used in other areas as well

## Py Descriptions ##
datagenerator.py: Is the script that generates the synthetic data
dataloader.py: Does the data processing for the model.
churn_model.py: This is the entry point to the code, I have copy pasted classes from the above scripts again, so that you do not have to reload the modules each time you the run this script.