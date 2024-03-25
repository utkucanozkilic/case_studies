import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings


warnings.filterwarnings('ignore')



# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# analysis_id: Bir scoutun bir oyuncunun değerlendirmesi için verdiği puanların toplamı
# attribute_id: Oyuncuların değerlendirildiği özelliklerin id'si. Örneğin; hız, pas, şut
# attribute_value: Bir scoutun bir oyuncunun bir özelliğine


def get_col_names(dataframe, cat_th = 10, car_th = 20, info = False):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['object', 'category', 'bool']]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtype in ['int', 'float']
                   and dataframe[col].nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtype in ['object', 'category']]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtype in ['float64', 'int64']]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    if not info:
        return cat_cols, num_cols, cat_but_car
    else:
        print("Observations: {}, Variables: {}".format(dataframe.shape[0], dataframe.shape[1]))
        print("Caterogical columns:", len(cat_cols))
        print("Numerical columns:", len(num_cols))
        print('Caterogical but cardinal columns:', len(cat_but_car))
        print('Numerical but caterogical columns:', len(num_but_cat))

        return cat_cols, num_cols, cat_but_car


pd.set_option('display.max_columns', 1881)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1881)

scoutium_df = pd.read_csv(
    r'/Machine Learning/Unsupervised Learning/talent_scouting/scoutium_attributes.csv'
    )

scoutium_labels_df = pd.read_csv(
    r'/Machine Learning/Unsupervised Learning/talent_scouting/scoutium_potential_labels.csv'
    )

# Adım 2:
df = pd.DataFrame.merge(
    scoutium_df, scoutium_labels_df,
    on = ['task_response_id', 'match_id', 'evaluator_id', 'player_id']
    )

# Adım 3:
df = df[~(df['position_id'] == 1)]

# Adım 4:
df = df[~(df['potential_label'] == 'below_average')]

# Adım 5:
df_pivot = pd.pivot_table(
    data = df,
    index = ['player_id', 'position_id', 'potential_label'],
    columns = 'attribute_id',
    values = 'attribute_value'
    )

df_pivot.reset_index(inplace = True)
df_pivot.columns = df_pivot.columns.astype('string')
df_pivot.columns.name = None

# Adım 6:
label_encoder = LabelEncoder()
# 0: average, 1: highlighted
df_pivot['potential_label'] = label_encoder.fit_transform(df_pivot['potential_label'])

# Adım 7:
cat_cols, num_cols, cat_but_car = get_col_names(df_pivot, cat_th = 5)

num_cols.remove('position_id')
num_cols.remove('player_id')

# Adım 8:
for col in num_cols:
    standart_scaler = StandardScaler()
    df_pivot[col] = standart_scaler.fit_transform(df_pivot[col].values.reshape(-1, 1))

# Adım 9:
X = df_pivot.drop(['position_id', 'player_id', 'potential_label'], axis = 1)
y = df_pivot['potential_label']


# n_neighbors: 5,
knn_param_grid = {
    'n_neighbors': [1, 3, 5]
    }

# penalty: l2, C: 1, solver: lbfgs
log_reg_param_grid = {
    'penalty': ['l1', 'l2'],
    'C':       [0.001, 0.01, 0.1, 1, 10, 100],  # Ters düzenleme parametresi
    'solver':  ['liblinear', 'saga', 'lbfgs']  # Optimizasyon algoritması
    }

# C: 1, kernel: rbf, gamma: scale
svc_param_grid = {
    'C':      [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma':  [0.001, 0.01, 0.1]
    }

# max_depth: None, min_samples_split: 2, min_samples_leaf: 1
cart_param_grid = {
    'max_depth':         [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4]
    }

# n_estimators: 100, max_depth: None, min_samples_split: 2, min_samples_leaf: 1, max_features: sqrt
rf_param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2']
    }

# n_estimators: 50, learning_rate: 1
adaboost_param_grid = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
    }

# n_estimators: 100, learning_rate: 0.1, max_depth: 3
gbm_param_grid = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth':     [3, 5, 7]
    }

# n_estimators: 100, learning_rate: 0.1, max_depth: -1, num_leaves: 31
lightgbm_param_grid = {
    'n_estimators':  [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth':     [3, 5, 7],
    'num_leaves':    [15, 31, 63],
    'verbosity': [-1]
    }

# iterations: 1000, learning_rate: 0.03, depth: 6
catboost_param_grid = {
    'iterations':    [50, 100, 200],
    'learning_rate': [0.01, 0.03, 0.1, 1.0],
    'depth':         [3, 5, 6]
    }

classifiers_param_grids = [
    ('knn_clf', KNeighborsClassifier(), knn_param_grid),
    ('log_reg', LogisticRegression(), log_reg_param_grid),
    ('svc_clf', SVC(), svc_param_grid),
    ('cart', DecisionTreeClassifier(), cart_param_grid),
    ('rf_clf', RandomForestClassifier(), rf_param_grid),
    ('adaboost', AdaBoostClassifier(), adaboost_param_grid),
    ('gbm', GradientBoostingClassifier(), gbm_param_grid),
    ('lightgbm', LGBMClassifier(), lightgbm_param_grid),
    ('catboost', CatBoostClassifier(verbose = False), catboost_param_grid)
    ]


def get_best_model(X, y, cv = 5, scoring = 'f1'):
    score = 0
    best_model_name = ''

    for name, model, params in classifiers_param_grids:
        grid_search_model = GridSearchCV(model, params, cv = cv, n_jobs = -1, verbose = False).fit(X, y)
        best_model = model.set_params(**grid_search_model.best_params_)

        cv_results = cross_validate(best_model, X, y, cv = cv, scoring = scoring, n_jobs = -1, verbose = False)
        print("##############  {}  ##############" .format(name.upper()))
        print("For '{}' metric, score: {} and best parameters: {}\n"
              .format(scoring, cv_results['test_score'].mean(), grid_search_model.best_params_))

        if cv_results['test_score'].mean() > score:
            best_model_name = name
            score = cv_results['test_score'].mean()

    return 'The best model is {}'.format(best_model_name.upper())


get_best_model(X, y, scoring = 'roc_auc')



