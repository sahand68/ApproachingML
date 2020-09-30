from sklearn import tree
from sklearn import ensemble 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
models = {

"decision_tree_gini":tree.DecisionTreeClassifier(
    criterion = "gini"
),
"decision_tree_entropy" : tree.DecisionTreeClassifier(

    criterion = "entropy"),

"rf": ensemble.RandomForestClassifier(),
"catboost":CatBoostClassifier(iterations=1000, 
                           task_type="GPU",
                           devices='0:1'),
"xgb": XGBClassifier(tree_method = 'gpu_hist'),
"lgbm":LGBMClassifier()


}