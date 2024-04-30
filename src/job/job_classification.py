import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


df_train = pd.read_csv("data/job_change/job_change_train.csv")
df_test = pd.read_csv("data/job_change/job_change_test.csv")

x_test = df_test
x_train = df_train.drop(columns=["willing_to_change_job"])

y_train = df_train["willing_to_change_job"]

y_train = y_train.replace({'Yes': 1, 'No': 0})

#non-numerical columns in the Dataframe
non_numeric_cols = x_train.select_dtypes(exclude=['number'])
print(non_numeric_cols.columns)

encoder = OrdinalEncoder()
encoded_train_data = encoder.fit_transform(x_train[non_numeric_cols.columns])
encoded_test_data = encoder.transform(x_test[non_numeric_cols.columns])

x_train[non_numeric_cols.columns] = encoded_train_data.astype(int)
x_test[non_numeric_cols.columns] = encoded_test_data.astype(int)


lf = LogisticRegression(random_state=0, solver="lbfgs", max_iter=2000, C=25).fit(x_train, y_train)
y_pred = lf.predict(x_test)

# Initialize and train Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(x_train, y_train)
rf_y_pred = rf_classifier.predict(x_test)

# Initialize and train Support Vector Machine (SVM) classifier
svm_classifier = SVC()
svm_classifier.fit(x_train, y_train)
svm_y_pred = svm_classifier.predict(x_test)

# Initialize and train Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(x_train, y_train)
gb_y_pred = gb_classifier.predict(x_test)
