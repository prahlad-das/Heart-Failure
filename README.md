# Heart-Failure

<p>This is a small data science project for beginners. Main goal of the project is to predict whether a patient will survive in case of heart attack. In our dataset we have information about patient's age, anaemia, creatinine, blood pressure, platelets etc. We have total 12 parameters of each patient. Our dataset is clean means we don't have missing values or NaN values in our dataset. We have splitted the dataset 80% for training and 20% for testing. </p>

We have tried different models for predictions. 
* Logistic Regression
* K Nearest Neighbors
* Support Vector Classification
* Naive Bayes
* Decision Tree
* Random Forest 
* XG Boost

```Python
def models(X_train, y_train):
    
    # logistic regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, y_train)
    
    # kNN
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    knn.fit(X_train, y_train)
    
    # SVC
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(X_train, y_train)
    
    # Gaussian Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(X_train, y_train)
    
    # Decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, y_train)
    
    # Random forest
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(X_train, y_train)
    
    # XGBoost
    from xgboost import XGBClassifier
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)

    
    # training accuracy
    print('[0] Logistic Regression training accuracy: ', log.score(X_train, y_train))
    print('[1] KNN training accuracy', knn.score(X_train, y_train))
    print('[2] SVC training accuracy', svc_rbf.score(X_train, y_train))
    print('[3] Naive Bayes training accuracy', gauss.score(X_train, y_train))
    print('[4] Tree training accuracy', tree.score(X_train, y_train))
    print('[5] Random forest training accuracy', forest.score(X_train, y_train))
    print('[6] XGBoost training accuracy', xgb.score(X_train, y_train))
    
    return log, knn, svc_rbf, gauss, tree, forest, xgb
```
<h3>Train accuracy for different models</h3>

Model | Train Accuracy
--------|-------------
Logistic regression train accuracy | 0.83
Nearest neighbor train accuracy | 0.73 
Support vector train accuracy | 0.67 
Naive Bayes train accuracy | 0.81 
Decision tree train accuracy | 1.0 
Random forest train accuracy | 0.98
XGBoost train accuracy | 1.0

<h3>Test accuracy for different models-</h3> 

Model | Test Accuracy
---- | ----
Logistic regression test accuracy | 0.76 
Nearest neighbor test accuracy | 0.68 
Support vector test accuracy | 0.7 
Naive Bayes test accuracy | 0.78 
Decision tree test accuracy | 0.8 
Random forest test accuracy | 0.76 
XGBoost test accuracy | 0.8 
