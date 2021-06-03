########## Class ##########

class Algorithm :
  
    # 초기화
    def __init__(self, df, *columns) :
        self.df = df

        cdf = self.df.copy()

        # column 선정
        drop_columns = list(columns)
        self.d_columns = drop_columns
        
        if len(columns) >= 1 :
            X_features = cdf.drop(drop_columns, axis=1)
            X_features.drop('LABEL', axis=1, inplace=True)
            y_label = cdf['LABEL']
        else : 
            X_features = cdf.drop('LABEL', axis=1)
            y_label = cdf['LABEL']
        
        # train_test_split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=0, stratify=y_label)

        self.X_features = X_features
        self.y_label = y_label
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    # 평가를 위한 사용자 함수 생성
    def clf_eval(self, y_test, preds) :

      from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        
      confusion = confusion_matrix(y_test, preds)
      accuracy = accuracy_score(y_test, preds)
      precision = precision_score(y_test, preds)
      recall = recall_score(y_test, preds)
      f1 = f1_score(y_test, preds)
        
      print('Confusion Matrix')
      print(confusion)
      print('Accuracy : {0:.4f}\nPrecision : {1:.4f}\nRecall : {2:.4f}\nf1 score : {3:.4f}'.format(accuracy, precision, recall, f1))


    # LightGBM
    def LGBM(self) :
        
        from lightgbm import LGBMClassifier
        import timeit

        start_time = timeit.default_timer()
        
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test

        lgb_wrapper = LGBMClassifier(n_estimators=100, boost_from_average=False)
        evals = [(X_test, y_test)]
        
        # 조기중단 수행
        lgb_wrapper.fit(X_train,y_train, early_stopping_rounds=10,eval_set=evals,
                        eval_metric='logloss')
        preds = lgb_wrapper.predict(X_test)
        
        # 평가 시행
        self.clf_eval(y_test, preds)
        
        # Feature Importance Visualization
        from lightgbm import plot_importance
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(13,25))
        plot_importance(lgb_wrapper, ax=ax)

        terminate_time = timeit.default_timer()

        print('time : {}'.format(terminate_time - start_time))
    

    # CatBoost
    def catboost(self) :

      # catboost는 사용시 아래 코드 미리 실행시켜 놓을 것
      # pip install catboost

      from catboost import CatBoostClassifier
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import seaborn as sns
      import timeit

      start_time = timeit.default_timer()

      X_train = self.X_train
      X_test = self.X_test
      y_train = self.y_train
      y_test = self.y_test
            
      cb_clf = CatBoostClassifier(iterations=100, logging_level ='Verbose')
      cb_clf.fit(X_train, y_train)
      cb_pred = cb_clf.predict(X_test)
      # cb_pred_proba=cb.predict_proba(X_test)[:,1]
      self.clf_eval(y_test, cb_pred)

      # Feature Importance 시각화
      feature_importances = cb_clf.get_feature_importance()
      feature_names = X_train.columns
      cb_importances = pd.Series(feature_importances, index = feature_names)
      cb_importances_sort = cb_importances.sort_values(ascending=False)
      plt.figure(figsize=(15,15))
      sns.barplot(x=cb_importances_sort, y = cb_importances_sort.index, color="#1F77B4")

      terminate_time = timeit.default_timer()

      print('time : {}'.format(terminate_time - start_time))

    # RandomForestClassifier
    def RF(self):

      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from sklearn.ensemble import RandomForestClassifier
      import timeit
      import seaborn as sns

      start_time = timeit.default_timer()

      X_train = self.X_train
      X_test = self.X_test
      y_train = self.y_train
      y_test = self.y_test

      rf_clf = RandomForestClassifier()
      rf_clf.fit(X_train, y_train)
      rf_pred = rf_clf.predict(X_test)
      # rf_pred_proba=rf.predict_proba(X_test)[:,1]
      self.clf_eval(y_test, rf_pred)

      # Feature Importance 시각화
      feature_importances = rf_clf.feature_importances_*100 # cb_feature_importances와 동일한 x축을 만들기 위해 *100을 함.
      feature_names = X_train.columns
      rf_importances = pd.Series(feature_importances, index = feature_names)
      rf_importances_sort = rf_importances.sort_values(ascending=False)
      plt.figure(figsize=(15,15))
      sns.barplot(x=rf_importances_sort, y = rf_importances_sort.index, color="#1F77B4")

      terminate_time = timeit.default_timer()

      print('time : {}'.format(terminate_time - start_time))



    # MLP
    def MLP(self) :

      import pandas as pd
      import numpy as np
      import timeit
      import matplotlib.pyplot as plt

      start_time = timeit.default_timer()

      X_features = self.X_features
      y_label = self.y_label

      # Standard Scaling
      from sklearn.preprocessing import StandardScaler
      ss = StandardScaler()
      scaled = ss.fit_transform(X_features)
      scaled_df = pd.DataFrame(scaled, columns = X_features.columns)
            
      # train_test_split (Scaled)
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(scaled_df, y_label, test_size=0.2, random_state=0, stratify=y_label)
      
      from sklearn.neural_network import MLPClassifier

      mlp = MLPClassifier()

      mlp.fit(X_train, y_train) 
      mlp_pred = mlp.predict(X_test)
      #mlp_pred_proba=mlp.predict_proba(X_test)[:,1]

      self.clf_eval(y_test, mlp_pred)

      terminate_time = timeit.default_timer()

      print('time : {}'.format(terminate_time - start_time)) 
    