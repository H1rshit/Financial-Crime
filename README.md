# Financial-Crime
Class imbalance and deciding the optimal threshold for classifier is a big challenge in this problem.

In the dataset we have three different files,
   fraudster.csv: containing users that have fraud transactions
   transactions.csv: containing all the transactions data
   users.csv: containing user level information
   
The notebook is self-explanatory and some key techniques used in this notebook are :
   SMOTE
   HyperParmeter Tuning
   Precision Recall Curve for optimal threshold
   Confusion Matrix
   
Once the model is trained, it is dumped as a pickle file

Patrol.py is the main file that is used to predict the labels for new transactions
