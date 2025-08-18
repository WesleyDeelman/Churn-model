### What is a churn model?

"A churn model is a predictive analytics tool that estimates the likelihood of customers discontinuing their relationship with a business within a given timeframe. It leverages historical data—such as usage patterns, engagement metrics, and demographic information—to identify behavioral signals that precede customer attrition. By flagging at-risk customers early, churn models empower companies to implement targeted retention strategies, reduce customer acquisition costs, and improve long-term profitability" according to co-pilot.

### CHURN MODEL
The aim of this repo is to demonstrate how to build a churn model, when there is only transactional data available i.e. TransactionID, CustomerID,Transaction Date, Transaction Amount. This scenario is in fact very common in practice, and although what we have here is built for a Churn Model, the features could be used in other areas as well

## Py Descriptions
**datagenerator.py:**   Is the script that generates the synthetic data
**dataloader.py:**   Does the data processing for the model.
**churn_model.py:**   This is the entry point to the code, I have copy pasted classes from the above scripts again, so that you do not have to reload the modules each time you the run this script.
