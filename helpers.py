from sklearn.base import BaseEstimator, TransformerMixin, clone
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error, accuracy_score, recall_score, auc
from sklearn.metrics import precision_score, confusion_matrix
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from multiprocessing import Pool
from sklearn.model_selection import KFold


class DropMissing(TransformerMixin, BaseEstimator):

    def __init__(self, cols_to_check, colnames, threshold=None):
        self.cols_to_check = cols_to_check
        self.colnames = colnames
        self.threshold = threshold
        self.columns_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.colnames)
        self.columns_ = X.columns
        missing = X[self.cols_to_check].isna().mean()
        self.missing_frac = missing
        self.cols_to_drop = missing[missing > self.threshold].index
        self.cols_to_keep = missing[missing <= self.threshold].index
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=self.colnames)
        #logger.info("Dropping columns with missingness over %s"%self.threshold)

        #if not set(X.columns) == set(self.columns_):
        #    logger.warning("Columns differ from in training.")

        return X.drop(self.cols_to_drop.values, axis=1)

    def get_feature_names(self):
        return list(set(self.columns_) - set(self.cols_to_drop))

class Winsorizer(TransformerMixin, BaseEstimator):
    def __init__(self, limits=None):
        self.limits = limits

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.limits is None:
            self.limits = (0.01, 0.99)
        elif isinstance(self.limits, float):
            self.limits = (self.limits, 1 - self.limits)

        columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        threshold_dict = {}

        for column in columns:
            low, high = X[column].quantile(self.limits)
            threshold_dict[column] = (low, high)

        self.columns_ = columns
        self.threshold_dict_ = threshold_dict

        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        X_t = X.copy()
        def trim(x, low, high):
            if pd.isna(x):
                return x
            else:
                x = low if x < low else x
                x = high if x > high else x
                return x
        trim_vec = np.vectorize(trim)

        for column, tup in self.threshold_dict_.items():
            X_t[column] = trim_vec(X_t[column], *tup)

        return X_t

    def get_feature_names(self, feature_in=None):
        return self.columns_
    
def clean_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tight_layout()
    
flatten = lambda l: [item for sublist in l for item in sublist]

EXCHANGE_RATE = 572.269

def fpr(x, y):
    tn, fp, fn, tp = confusion_matrix(x, y).ravel()
    return fp/(fp + tn)

def tpr(x, y):
    tn, fp, fn, tp = confusion_matrix(x, y).ravel()
    return tp/(tp + fn)

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def metrics(a1, a2, p1, p2):
    
    if p1 == 0 or p2 == 0 or p1 == 100 or p2 == 100:
        raise ValueError('Pecentage targeting must be between 0 and 100 (exclusive).')

    num_ones = int((p1/100)*len(a1))
    num_zeros = len(a1) - num_ones
    targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
    a = np.vstack([a1, a2])
    a = a[:, a[0, :].argsort()]
    a[0, :] = targeting_vector
    
    np.random.seed(12)
    a = a[:, np.random.rand(a.shape[1]).argsort()]
    
    num_ones = int((p2/100)*len(a2))
    num_zeros = len(a2) - num_ones
    targeting_vector = np.concatenate([np.ones(num_ones), np.zeros(num_zeros)])
    a = a[:, a[1, :].argsort()]
    a[1, :] = targeting_vector
    
    tn, fp, fn, tp = confusion_matrix(a[0, :], a[1, :]).ravel()

    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    tpr = recall
    fpr = fp/(fp + tn)
    return accuracy, precision, recall, tpr, fpr

def auc_overall(a1, a2):
    grid = np.linspace(1, 100, 99)[:-1]
    metrics_grid = [metrics(a1, a2, p, p) for p in grid]
    tprs, fprs = [g[3] for g in metrics_grid], [g[4] for g in metrics_grid]
        
    fprs = [0] + fprs + [1]
    tprs = [0] + tprs + [1]
    
    while not strictly_increasing(fprs):
        to_remove = []
        for j in range(1, len(fprs)):
            if fprs[j] <= fprs[j-1]:
                to_remove.append(j)
        fprs = [fprs[i] for i in range(len(fprs)) if i not in to_remove]
        tprs = [tprs[i] for i in range(len(tprs)) if i not in to_remove]
    return fprs, tprs, auc(fprs, tprs)

def table(args):
    
    df, groundtruth, proxies, proxynames, p1, p2, round_numbers = args
    np.random.seed(9)
    table = []
    
    df = df.dropna(subset=proxies)
    if round_numbers:
        print('Number of observations: %i (%i unique)' % (len(df), len(df['uid'].unique())))
    df[groundtruth] = df[groundtruth].astype('float')
          
    for p, proxy in enumerate(proxies):
          
        df[proxy] = df[proxy].astype('float')
        accuracy, precision, recall, _, _ = metrics(df[groundtruth], df[proxy], p1, p2)
        #if proxy == 'formal_occupation':
        #    spearman, auc_score = np.nan, np.nan
        if True:
            spearman = spearmanr(df[groundtruth], df[proxy])[0]
            _, _, auc_score = auc_overall(df[groundtruth].values, df[proxy].values)
        table.append([proxynames[p], spearman, auc_score, accuracy, precision, recall])
          
    table = pd.DataFrame(table)
    table.columns = ['Targeting Method', 'Spearman', 'AUC', 'Accuracy', 'Precision', 'Recall']
    if round_numbers:
        table = table.round(2)
        
    return table
  
def std_table(args):
    
    bootstraps, groundtruth, proxies, proxynames, p1, p2, round_numbers=args

    pool = Pool(56)
    args = [(bootstraps[i], groundtruth, proxies, proxynames, p1, p2, False) for i in range(len(bootstraps))]
    bootstraps = pool.map(table, args)
    pool.close()
    std = pd.concat(bootstraps, axis=0).groupby('Targeting Method').agg('std')
    sorting = {proxy:p for p, proxy in enumerate(proxynames)}
    std['Targeting Method'] = std.index
    std['key'] =std['Targeting Method'].apply(lambda x: sorting[x])
    std = std.sort_values('key', ascending=True)\
        [['Targeting Method', 'Spearman', 'AUC', 'Accuracy', 'Precision', 'Recall']]
    if round_numbers:
        std = std.round(4)
    
    return std

def get_crra(df, outcomes, outcomenames, groundtruth, budget, cashout_fee):
    df = df.dropna(subset=outcomes)
    grid = np.linspace(1, len(df), 100)
    curves = {}
    for o, outcome in enumerate(outcomes):
        df = df.sort_values(outcome, ascending=True)
        utilities, percent_targeted, transfersizes = [], [], []
        for i in grid:
            num_targeted = int(i)
            transfersizes.append((budget/num_targeted)*(30))
            df['targeted'] = np.concatenate([np.ones(num_targeted), np.zeros(len(df) - num_targeted)])
            percent_targeted.append(df['targeted'].mean())
            df['benefits'] = df['targeted']*(budget/num_targeted - cashout_fee)
            rho = 3
            df['utility'] = ((df[groundtruth] + df['benefits'])**(1-rho))/(1-rho)
            utilities.append(df['utility'].sum())
        curves[outcomenames[o]] = (percent_targeted, transfersizes, utilities)
    return curves

def inmultiple(lst1, lst2):
    '''
    Get all matches to a partial substring in a list. Used to find dummies related to a given feature.
    '''
    allmatches = []
    for x in lst1:
        allmatches = allmatches + [y for y in lst2 if x in y]
    return allmatches

def cv(model, x, y, weights):
    '''
    Cross-validation for sklearn models with built-in weighting (linear regression, ridge, random forest, xgboost)
    '''
    x = x.values
    kf = KFold(n_splits=5, shuffle=True, random_state=10)
    train_scores, test_scores = [], []
    for train_idx, test_idx in kf.split(x):
        # Get train and test splits for this fold for x, y, and weights
        x_train_fold, x_test_fold = x[train_idx], x[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        weights_train_fold, weights_test_fold = weights[train_idx], weights[test_idx]
        # Fit model with weights
        model.fit(x_train_fold, y_train_fold, sample_weight=weights_train_fold)
        # Get predictions 
        yhat_train_fold = model.predict(x_train_fold)
        yhat_test_fold = model.predict(x_test_fold)
        # Calculate weighted scores
        train_scores.append(r2_score(y_train_fold, yhat_train_fold, sample_weight=weights_train_fold))
        test_scores.append(r2_score(y_test_fold, yhat_test_fold, sample_weight=weights_test_fold))
    return np.mean(train_scores), np.mean(test_scores)

def get_bootstraps(df, n_bootstraps=1000):
    bootstraps = []
    for i in range(n_bootstraps):
        sample = df.sample(n=len(df), replace=True)
        sample = pd.DataFrame(np.repeat(sample.values, sample['weight'], axis=0), columns=sample.columns)
        bootstraps.append(sample)
    
    return bootstraps