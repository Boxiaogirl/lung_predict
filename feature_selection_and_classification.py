import os
import pandas as pd
import time
import joblib
from IPython.display import display
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from onekey_algo import OnekeyDS as okds
from onekey_algo.custom.utils import print_join_info
from code_package.comp1 import normalize_df,select_feature,lasso_cv_coefs,create_clf_model,plot_feature_importance, smote_resample 
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from onekey_algo.custom.components.delong import calc_95_CI
from onekey_algo.custom.components.metrics import analysis_pred_binary

def main():
   #os.makedirs('img', exist_ok=True)
   #os.makedirs('results', exist_ok=True)
   #os.makedirs('features', exist_ok=True)

   # the directory for compressed deep features (csv file)
   feature_file = r'./compress_features.csv'
   # the directory for label file (csv file)
   label_file = r'./label.csv'
   # read the name of label column
   labels = ['label']

   feature_data = pd.read_csv(feature_file)
   label_data = pd.read_csv(label_file)

   #print_join_info(feature_data, label_data)
   combined_data = pd.merge(feature_data, label_data, on=['ID'], how='inner')
   ids = combined_data['ID']
   combined_data = combined_data.drop(['ID'], axis=1)
   #print(combined_data[labels].value_counts())
   combined_data.describe()
   #print(combined_data.columns)
   
   #normalize feature to distribution mean=0, std=1
   data = normalize_df(combined_data, not_norm=labels, group='group')
   data = data.dropna(axis=1)
   #print(data.describe())
   
   pearson_corr = data[data['group'] == 'train'][[c for c in data.columns if c not in labels and c not in ['group']]].corr('pearson')
   sel_feature = select_feature(pearson_corr, threshold=0.9, topn=32, verbose=False)
   sel_feature = sel_feature + labels + ['group']

   sel_data = data[sel_feature]
   #sel_data.to_csv('selection_features.csv', header=True, index=False)

   n_classes = 2
   train_data = sel_data[(sel_data['group'] == 'train')]
   train_ids = ids[train_data.index]
   train_data = train_data.reset_index()
   train_data = train_data.drop('index', axis=1)
   y_data = train_data[labels]
   x_data = train_data.drop(labels + ['group'], axis=1)

   test_data = sel_data[sel_data['group'] != 'train']
   test_ids = ids[test_data.index]
   test_data = test_data.reset_index()
   test_data = test_data.drop('index', axis=1)
   y_test_data = test_data[labels]
   x_test_data = test_data.drop(labels + ['group'], axis=1)

   y_all_data = sel_data[labels]
   x_all_data = sel_data.drop(labels + ['group'], axis=1)

   column_names = x_data.columns
   print(f"training sample：{x_data.shape}, test sample：{x_test_data.shape}")

   alpha = lasso_cv_coefs(x_data, y_data, column_names=None, alpha_logmin=-3)

   models = []
   for label in labels:
       clf = linear_model.Lasso(alpha=alpha)
       clf.fit(x_data, y_data[label])
       models.append(clf)

   COEF_THRESHOLD = 1e-8 # threshold for selecting
   scores = []
   selected_features = []
   for label, model in zip(labels, models):
       feat_coef = [(feat_name, coef) for feat_name, coef in zip(column_names, model.coef_) 
                    if COEF_THRESHOLD is None or abs(coef) > COEF_THRESHOLD]
       selected_features.append([feat for feat, _ in feat_coef])
       formula = ' '.join([f"{coef:+.6f} * {feat_name}" for feat_name, coef in feat_coef])
       score = f"{label} = {model.intercept_} {'+' if formula[0] != '-' else ''} {formula}"
       scores.append(score)

   feat_coef = sorted(feat_coef, key=lambda x: x[1])
   feat_coef_df = pd.DataFrame(feat_coef, columns=['feature_name', 'Coefficients'])
   #feat_coef_df.plot(x='feature_name', y='Coefficients', kind='barh')

   x_data = x_data[selected_features[0]]
   x_test_data = x_test_data[selected_features[0]]
   x_data.columns

   model_names = ['LR']#, 'KNN', 'RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'NaiveBayes', 'AdaBoost', 'GradientBoosting', 'LR', 'MLP'
   models = create_clf_model(model_names)
   model_names = list(models.keys())

   x_train_sel, x_test_sel, y_train_sel, y_test_sel = x_data, x_test_data, y_data, y_test_data

   targets = []
   os.makedirs('classifier_models', exist_ok=True)
   for l in labels:
       new_models = list(create_clf_model(model_names).values())#okcomp.comp1.
       for mn, m in zip(model_names, new_models):
           x_train_smote, y_train_smote = x_train_sel, y_train_sel
           #x_train_smote, y_train_smote = smote_resample(x_train_sel, y_train_sel)
           print('Start training classifier model.............................. ')
           m.fit(x_train_smote, y_train_smote[l])
           # 保存训练的模型
           joblib.dump(m, f'classifier_models/{mn}_{l}.pkl') 
       targets.append(new_models)

   predictions = [[(model.predict(x_train_sel), model.predict(x_test_sel)) 
                for model in target] for label, target in zip(labels, targets)]
   pred_scores = [[(model.predict_proba(x_train_sel), model.predict_proba(x_test_sel)) 
                for model in target] for label, target in zip(labels, targets)]

   metric = []
   pred_sel_idx = []
   for label, prediction, scores in zip(labels, predictions, pred_scores):
       pred_sel_idx_label = []
       for mname, (train_pred, test_pred), (train_score, test_score) in zip(model_names, prediction, scores):
           #calculate metics on training data
           acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_train_sel[label], 
                                                                                              train_score[:, 1])
           ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
           metric.append((mname, acc, auc, ci, tpr, tnr, f"{label}-train"))# ppv, npv, precision, recall, f1, thres,
                 
           #calculate metrics on testing data
           acc, auc, ci, tpr, tnr, ppv, npv, precision, recall, f1, thres = analysis_pred_binary(y_test_sel[label], 
                                                                                              test_score[:, 1])
           ci = f"{ci[0]:.4f} - {ci[1]:.4f}"
           metric.append((mname, acc, auc, ci, tpr, tnr, f"{label}-test"))# ppv, npv, precision, recall, f1, thres,

           #calculate the thres corresponding to the sel idx
           pred_sel_idx_label.append(np.logical_or(test_score[:, 0] >= thres, test_score[:, 1] >= thres))
    
       pred_sel_idx.append(pred_sel_idx_label)
   metric = pd.DataFrame(metric, index=None, columns=['model_name', 'Accuracy', 'AUC', '95% CI','Sensitivity', 'Specificity', 'Task'])#'PPV', 'NPV', 'Precision', 'Recall', 'F1','Threshold', 
   print(metric)


if __name__ == '__main__':
   start = time.time()
   main()
   end = time.time()
   print(end - start)







