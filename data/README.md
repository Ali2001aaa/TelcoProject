# Data

The IBM Telco Customer Churn CSV is **not committed** to this repository: I could not find an explicit license
for the data file (the source repository's Apache-2.0 license covers its code pattern). Please credit IBM.

The notebook downloads it automatically from:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Manual download (if the notebook cannot reach the internet):
```bash
curl -L -o data/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```
The notebook falls back to `data/Telco-Customer-Churn.csv` if the URL fails (the file is git-ignored).

Source: IBM Cognos Analytics sample data, fictional telecom company.
