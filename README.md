# Financial-Crime
Class imbalance and deciding the optimal threshold for classifier is a big challenge in this problem.

In the dataset we have three different files,
   1) fraudster.csv: containing users that have fraud transactions
   2) transactions.csv: containing all the transactions data
   3) users.csv: containing user level information
   
The notebook is self-explanatory and some key techniques used in this notebook are :
   1) SMOTE
   2) HyperParmeter Tuning
   3) Precision Recall Curve for optimal threshold
   4) Confusion Matrix
   
Once the model is trained, it is dumped as a pickle file

Patrol.py is the main file that is used to predict the labels for new transactions
