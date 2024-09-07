"""
* This file is copyright (c) 2024 Srikumar Krishnamoorthy
* 
* This program is free software: you can redistribute it and/or modify it under the
* terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
* A PARTICULAR PURPOSE. See the GNU General Public License for more details.
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*
"""

from sklearn import datasets
import pandas as pd, numpy as np, time, os, csv, operator, datatable as dt, copy, scipy
from sklearn.datasets import fetch_california_housing

from sklearn import tree, model_selection, metrics
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import make_scorer
from hmeasure import h_score

from sklearn.inspection import permutation_importance
import seaborn as sns    
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings("ignore")
import random
import dmba

from IPython.display import display, HTML
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge

###################################################read file utilities##########################################################
class DataUtils:
    def __init__(self):
        pass
        
    def get_dataset_df(self, params):
        dsName = params.get('dsName')
        taskType = params.get('taskType', 'classification')

        if dsName=='Iris':
            iris = datasets.load_iris()
            X = iris.data 
            y = iris.target
    
            data = pd.DataFrame(X, columns=iris.feature_names)
            data.columns = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
            X = data
            
            numericColumns = data.columns.tolist()
            catColumns = []
    
            numericIntCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, np.integer)]
            numericFloatCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, float)]
            catCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]

        elif dsName=='BankMarketingUCI':
            fname = 'datasets/bank marketing.csv'
            data = pd.read_csv(fname, sep=';')
            for col in data.columns.tolist():
                missingEntriesByCol = len(data[col][data[col]=="unknown"].index) 
                if missingEntriesByCol > 0:
                    mcommonVal = data[col].mode()[0]
                    if col=='default': mcommonVal = 'no'#can't set it to unknown which is the most common value
                    data[col] = data[col].apply(lambda x: mcommonVal if x=="unknown" else x)
            
            X, y = data.iloc[:,:-1], data.iloc[:,-1]
            numericColumns = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']#skip duration column as per the dataset details
            catColumns = ['job','marital','education','default','housing','loan']#poutcome, contact with large % of unknown (or null) values dropped; dropping 'day, month columns - just retaining pdays column'
            numericIntCols = ['age']
            numericFloatCols = ['campaign', 'previous', 'pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
        
        elif dsName=='pimaIndianDiabetes':
            fname='datasets/pima indians diabetes.csv'
            data = pd.read_csv(fname, header=None)
            data.columns = ['numPregnancies', 'glucose', 'bp', 'skinThickness', 'insulin', 'bmi', 'diabetesPedigreeFunction', 'age', 'class']
            X, y = data.iloc[:,:-1], data.iloc[:,-1]
            numericColumns, catColumns = X.columns.tolist(), []
    
            numericIntCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, np.integer)]
            numericFloatCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, float)]
            catCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]
        
        elif dsName=='Heloc':
            fname = 'datasets/heloc.csv'
            data = pd.read_csv(fname)
        
            #exclude records with missing score values
            data = data[data['ExternalRiskEstimate']!=-9]#missing values
            for c in data.columns.tolist()[2:]:#replace with median
                col_med = data.loc[:,c].median()
                data.loc[:, c].replace([-7, -8], [col_med, col_med], inplace=True)
    
            dataTmp = data.iloc[:, 1:]
            negValueColumns = dataTmp.columns[(dataTmp < 0).any()].tolist()
            if len(negValueColumns)>0:
                print('columns with negative values ', negValueColumns)
            
            #X, y = data.iloc[:, 2:], data.iloc[:, 0] #external risk estimate has perfect correlation with target
            X, y = data.iloc[:, 1:], data.iloc[:, 0] #using external risk estimate - a dominant attribute (similar results on both scenarios; but the feature importance dramatically changes for this 1 variable which is dependent on others)
            X = X.apply(pd.to_numeric).astype('float')#convert to float to apply scalar operations
            numericColumns, catColumns = X.columns.tolist(), []
    
            numericIntCols = []
            numericFloatCols = list(set(numericColumns) - set(numericIntCols))
    
        elif dsName=='Titanic':
            fname = 'datasets/titanic.csv'
            data = pd.read_csv(fname)
            
            most_common_value = data['Embarked'].mode()[0]
            data['Embarked'] = data['Embarked'].fillna(most_common_value)
            data['Age'] = data['Age'].fillna(data['Age'].mean())
            data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
            data['Fare'] = np.log(data['Fare']+1)
            
            data = data.assign(numRelatives = data[['SibSp', 'Parch']].apply(lambda x: x['SibSp']+x['Parch'], axis=1))
    
            X, y = data[['Pclass', 'Age', 'Sex', 'numRelatives', 'Fare', 'Embarked']], data['Survived']
            X = X.assign(Pclass=X['Pclass'].apply(str)) #convert numeric to object/string type; otherwise dummy coding doesn't work
            numericColumns, catColumns = ['Age', 'Fare', 'numRelatives'], ['Pclass', 'Sex', 'Embarked']
            numericIntCols = [colx for colx in numericColumns if np.issubdtype(X[colx].dtype, np.integer)]
            numericFloatCols = [colx for colx in numericColumns if np.issubdtype(X[colx].dtype, float)]
            #numericColumns, binaryColumns, catColumns = ['Pclass', 'Age', 'numRelatives', 'Fare'], [], ['Sex', 'Embarked']

        else:#reading by file name, ensure that the file is placed in the datasets folder and the last column is the target column (or specify targetColumn parameter)
            print("dsName not found, reading by file name (use the last column as the target variable or the one specified in targetColumn) ") #read by file name specified and infer data type 
    
            if params.get('fileName', -1)==-1:
                print('No file name specified, exiting ')
                return None, None, None, None
            
            fname='datasets/'+params['fileName']
            if not os.path.isfile(fname):
                print('File '+fname+' not found in the directory ')
                return None, None, None, None 
                
            fHeader = params.get('fileHeader', True)
            fSep = params.get('fileSep', ',')
            nrows = params.get('nrows', -1)#sample read
            if '.csv' in fname or '.data' in fname or '.zip' in fname or '.gz' in fname or '.gzip' in fname:
                #.data file assumed to be in csv format, .zip assumed to have a csv format data (.zip has to be done using 7z)
                if fHeader:#first row has column names
                    if nrows==-1: 
                        data = pd.read_csv(fname, sep=fSep)
                    else: 
                        data = pd.read_csv(fname, sep=fSep, nrows=nrows)
                else: 
                    if nrows==-1: data = pd.read_csv(fname, header=None, sep=fSep)
                    else: data = pd.read_csv(fname, header=None, sep=fSep, nrows=nrows)
            elif '.xlsx' in fname:
                data = pd.read_excel(fname, header=fHeader)
            else:
                print('unrecognized file extension, use .csv or .data (csv) or .xlsx files ')
                return None, None, None, None
    
            columnsToDrop = params.get('dropColumns', [])
            if len(columnsToDrop)>0:
                data = data.drop(columnsToDrop, axis=1)
    
            #remove columns with large number of missing values
            miss =data.isna().sum().sort_values(ascending=False).reset_index(drop=False).rename({0:'missing_value_count'}, axis=1)
            miss['missing_value_ratio'] = (miss['missing_value_count'] / data.shape[0]) * 100
            colsWithLargeMissingValues = miss[miss['missing_value_ratio']>60]['index'].tolist()
            data = data.drop(columns = colsWithLargeMissingValues)
    
            if params.get('dtypeObject', False):#force all columns to object or categorical type 
                for c in data.columns:
                    data[c] = data[c].astype(str)
            
            targetColumn = params.get('targetColumn', -1)#target column specified
            if targetColumn==-1:
                X, y = data.iloc[:,:-1], data.iloc[:,-1]
            else:
                print('target column specified', targetColumn)
                if not np.issubdtype(data[targetColumn].dtype, object):
                    data[targetColumn] = data[targetColumn].astype(str)
                #target column has to be categorical, if it is numerical 0 or 1, change it to categorical type
                X, y = data.drop(targetColumn, axis=1), data[targetColumn]
    
            X.columns = [str(c) for c in X.columns.tolist()]
            numericColumns = [colx for colx in X.columns.tolist() if not np.issubdtype(X[colx].dtype, object)]
            catColumns = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]
    
            if params.get('condenseCatColumns', False):#5 + 1 other category transformation
                Xcat = pd.DataFrame(self.condenseCategoricalColumns(X[catColumns].to_numpy()), columns=X[catColumns].columns)
                X = pd.concat([X[numericColumns], Xcat], axis=1)
            
            #miss value imputation
            for c in catColumns:
                most_common_value = X[c].mode()[0]
                X[c] = X[c].fillna(most_common_value)
            for c in numericColumns:
                X[c] = X[c].fillna(X[c].median())
            
            numericIntCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, np.integer)]
            numericFloatCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, float)]
            catCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]
        
        allCols = [numericIntCols, numericFloatCols, catColumns]
        if taskType=='classification':
            u, cnts = np.unique(y, return_counts=True)
            rx = sorted([(a, b) for a, b in zip(u, cnts)], key=lambda x: -x[1])
            mapYbyCntsDesc = dict([(rxi[0], ridx) for ridx, rxi in enumerate(rx)])
            y = pd.Series(y).map(mapYbyCntsDesc).to_numpy() #transformed y - higher value assigned to class with smaller number of examples
            yNewToOriginal = mapYbyCntsDesc
        else: y = y.astype('float32')
            
        X = X[[ai for a in allCols for ai in a]]
        X = X.reset_index(drop=True)
    
        procdata = {'allCols': allCols, 'origColumns': X.columns.tolist()}
        self.displayInfo(X, y, procdata, params)
    
        if taskType=='classification':
            return X, y, yNewToOriginal, procdata
        else: return X, y, procdata
    
    def displayInfo(self, X, y, procdata, params):
        printParamSet = ['B', 'L', 'G']
        taskType = params.get('taskType', 'classification')
    
        if taskType=='classification':
            yvals = np.unique(y, return_counts=True)
        featureSizes = (len(procdata['allCols'][0]), len(procdata['allCols'][1]), len(procdata['allCols'][2]))
    
        if taskType=='classification':
            print("dataset:", params.get('dsName'), X.shape, ' featureSize:', featureSizes, 'classSize:', list(zip(yvals[0], yvals[1])))
        else: print("dataset:", params.get('dsName'), X.shape, ' featureSize:', featureSizes)
        
        if params.get('verbose', True):
            print('all cols:',[a[0:15] for a in X.columns.tolist()[0:10]])
            print('i/f cols:',procdata['allCols'][0][0:10], procdata['allCols'][1][0:10])
            print('cat cols:',procdata['allCols'][2][0:10])
        print("params  :", [(k, v) for k, v in params.items() if k in printParamSet]) 
        
    def _condenseCategoricalColumns(self, arrn, N=5, returnNlargest=False):#one column
        nval = min(len(arrn), N)
        values, counts = np.unique(arrn, return_counts=True)
        nlargest = values[counts.argsort()][::-1][:nval]
        if returnNlargest: return nlargest
        else: return np.where(np.in1d(arrn, nlargest), arrn, 'Other')
    
    def condenseCategoricalColumns(self, arrn, N=5, returnNlargest=False):#entire array
        return np.apply_along_axis(self._condenseCategoricalColumns, 0, arrn, N, returnNlargest)

#############################################metric utilities#########################################################################

class MetricUtils:
    def __init__(self):
        self.logLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)
        self.hmeasure = make_scorer(h_score, greater_is_better=True, needs_proba=True)
        self.roc_auc_mclass = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
        self.hmeasure_mclass = make_scorer(self.h_score_multiclass, greater_is_better=True, needs_proba=True)
        self.precMacro = make_scorer(precision_score, average="macro",zero_division=0)
        self.recallMacro = make_scorer(recall_score, average="macro",zero_division=0)
        self.f1Macro = make_scorer(f1_score, average="macro",zero_division=0)
        
        self.rmsle = make_scorer(self.RMSLE, greater_is_better=False)
 
    def h_score_multiclass(self, y_test, y_pred_proba):
        hm = []
        for i in np.unique(y_test):
            y_pred_i = y_pred_proba[:, i]#extract for each class
            y_test_i = copy.deepcopy(y_test)
            y_test_i[y_test_i==i] = 333 #class of interest
            y_test_i[y_test_i!=333] = 444 #other classes
            y_test_i[y_test_i==333] = 1 #class of interest label 1
            y_test_i[y_test_i==444] = 0 #other classes, label 0
            hmeasure = h_score(y_test_i, y_pred_i)#one-vs-rest
            hm.append(hmeasure)
        
        return round(np.mean(hm), 4)

    def getScorer(self, y):
        if len(np.unique(y))==2: 
            binaryClassScorer = {'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall', 
                                  'f1': 'f1',  'auc': 'roc_auc', 'logLoss': self.logLoss, 'hmeasure': self.hmeasure}
            return binaryClassScorer
        else:#multi-class
            multiClassScorer = {'accuracy': 'accuracy', 'precision': self.precMacro, 'recall': self.recallMacro, 
                                 'f1': self.f1Macro,  'auc': self.roc_auc_mclass, 'logLoss': self.logLoss, 'hmeasure': self.hmeasure_mclass}
            return multiClassScorer

    def RMSLE(self, y_test, y_pred):#root mean squared log error
        return np.sqrt(mean_squared_log_error(y_test, y_pred))

    def getScorerRegression(self):
        return ('r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error')

    def get_metrics_mclass(self, y_test, y_pred, y_pred_proba):#multi-class; macro averaged values
        enc = OneHotEncoder(categories=[list(range(y_pred_proba.shape[1]))], sparse=False)#dummy coding might not handle all missing classes correctly, though pred_proba is generated for all classes
        y_test_enc = enc.fit_transform([[yi] for yi in y_test]).astype('int')
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test_enc, y_pred_proba, multi_class='ovr')
        p, r, f = precision_score(y_test, y_pred, average="macro",zero_division=0), recall_score(y_test, y_pred, average="macro",zero_division=0), f1_score(y_test, y_pred, average="macro",zero_division=0)
        hscore = self.h_score_multiclass(y_test, y_pred_proba)
        lgLossLbls = range(y_pred_proba.shape[1])
        lgLoss = round(log_loss(y_test_enc, y_pred_proba), 4)
        return [round(accuracy, 4), round(f, 4), round(auc, 4), hscore, lgLoss, round(p, 4), round(r, 4)]#acc, f1, auc, hmeasure, prec, recall

    def get_metrics(self, y_test, y_pred, y_pred_proba):#binary classification
        if y_pred_proba.shape[1]>2: return self.get_metrics_mclass(y_test, y_pred, y_pred_proba)
        prfs = precision_recall_fscore_support(y_test, y_pred, warn_for=tuple())
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])#e.g. class [0, 1]
        hscore = round(h_score(y_test, y_pred_proba[:, 1]), 4)
        lgLoss = round(log_loss(y_test, y_pred_proba), 4)
        return [round(accuracy, 4), round(prfs[2][1], 4), round(auc, 4), hscore, lgLoss, round(prfs[0][1], 4), round(prfs[1][1], 4)]#acc, f1, auc, prec, recall

    def get_metrics_reg(self, y_test, y_pred):
        r2 = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        
        return [r2, mae, mse, rmse, mape]

    def display_final_scores(self, r):#input is output from cross_validate function with metrics acc, prec, recall, f1, auc, logLOss
        out = {}
        outKeys = ['test_'+k for k in ['accuracy', 'f1', 'auc', 'hmeasure', 'logLoss']] 
        for k, v in r.items():
            if not k in outKeys: continue
            if k=='test_logLoss':
                out[k.split('_')[1]] = -round(v.mean(), 2)
            else:
                out[k.split('_')[1]] = round(v.mean(), 2)
        
        res = pd.DataFrame(out.values(), index=out.keys()).T
        res.index = ['performance']
        res[[c.split('_')[1] for c in outKeys]]
        display(res)

    def display_final_scores_regression(self, r):#input is output from cross_validate function with metrics 
                                            #('r2', 'neg_mean_absolute_error' 'neg_mean_squared_error', 'neg_root_mean_squared_error', 
                                                                                #'neg_mean_absolute_percentage_error')
        out = {}
        outKeys = ['test_'+k for k in self.getScorerRegression()] 
        for k, v in r.items():
            if not k in outKeys: continue
            if k=='test_r2':
                out[k.split('test_')[1]] = round(v.mean(), 2)
            else:
                out[k.split('test_neg_')[1]] = -round(v.mean(), 2)
        
        res = pd.DataFrame(out.values(), index=out.keys()).T
        res.index = ['performance']
        display(res)

########################################plot utilities####################################################################
#1d plot
class PlotUtils:
    def __init__(self):
        pass
        
    def plot_relplot(self, clf, sampleSz=30, scale=0.5):#get samples from each class with high probability
        random.seed(1729)
        a, b = clf.x_test_hup_.todense(), clf.y_pred_
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        
        figsize=(.1*len(df.columns)*scale, 0.2*df.shape[1]*scale) #row size, col size
        cvals = list(set(clf.y_pred_))#distinct class values
        datFinal = df
        if df.shape[0]>sampleSz:#get samples from each class with high probability
            s = int(round(sampleSz/len(cvals), 0))
            for ci, classNo in enumerate(cvals):#for each class
                prob = clf.y_pred_proba_.T[classNo]
                dfx = pd.concat([df, pd.DataFrame(prob, columns=['prob'])], axis=1)
                dfx1 = dfx.sort_values(by=['prob'], ascending=False)
                
                if ci==0:
                    datFinal = dfx1.iloc[:s,:]#top part
                else: datFinal = pd.concat([datFinal, dfx1.iloc[:s,:]], axis=0)#picking the ones with top probability values for that class
    
        dat = datFinal.sort_values(by=['label'])#sort by label
        
        if 'prob' in dat.columns:
            dat = dat.iloc[:, :-1]#discard the last prob column
        dat = dat[sorted(dat.columns[:-1])+['label']]#sort by column
        
        dcntGrouped = dat.groupby('label').sum().T
        dcntGrouped = dcntGrouped.reset_index()
        dcntGrouped = dcntGrouped.sort_values(by=list(range(len(cvals))))
        
        dcnt = dict((dat.sum(axis=0)))
        dat = dat[dcntGrouped['index'].tolist()+['label']]
        
        dat = dat[dat.columns].reset_index(drop=True).reset_index()
        dfFlattened = dat.melt(id_vars=['index', 'label'], value_vars=dat.columns.tolist())
        dfFlattened.columns = ['idx', 'label', 'patterns', 'value']
        dfFlattened = dfFlattened.sort_values(by=['label', 'patterns'])
        
        dx = dfFlattened[(dfFlattened['value']>0)]
        g = sns.relplot(x="idx", y="patterns", hue='label', style='label',
                                data=dx, s=15, palette=sns.color_palette("tab10"), height=figsize[0]*2, aspect=1.5)
        axes = g.axes.flat[0]
        axes.set_ylabel('')
        axes.set_xlabel('data instances')
        axes.yaxis.set_tick_params(labelsize=14)
        axes.xaxis.set_tick_params(labelsize=9, rotation=90)
        #axes.get_xaxis().set_visible(False)
        px=axes.get_xaxis().set_ticks([])
    
    
    def plot_relplot_byclass(self, clf, classNo=1, sampleSz=30, scale=0.5):
        random.seed(1729)
        a, b = clf.x_test_hup_.todense(), clf.y_pred_
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        df = df[df.label==classNo]
        
        figsize=(.1*len(df.columns)*scale, 0.2*df.shape[1]*scale) #row size, col size
        cvals = list(set(clf.y_pred_))
        datFinal = df
        if df.shape[0]>sampleSz:#get samples from each class with high probability
            s = sampleSz        
            prob = clf.y_pred_proba_.T[classNo]
            dfx = pd.concat([df, pd.DataFrame(prob, columns=['prob'])], axis=1)
            dfx1 = dfx.sort_values(by=['prob'], ascending=False)
            dat = dfx1#.iloc[:, :-1]#discard the last prob column
            datFinal = dat.iloc[:s,:]#top part
            
        dat = datFinal.sort_values(by=['label'])#sort by label
        if 'prob' in dat.columns:
            dat = dat.iloc[:, :-1]#discard the last prob column
        dat = dat[sorted(dat.columns[:-1])+['label']]#sort by column
        
        datt = dat.T
        dat = datt[(datt.T != 0).any()].T #drop all zero cases
        if classNo==0:#zero label gets removed for 0 cases
            dat['label'] = [0]*dat.shape[0]
        dcntGrouped = dat
        dcntGrouped = dcntGrouped.reset_index()
        dcnt = dict((dat.sum(axis=0)))
    
        dat = dat[dat.columns].reset_index(drop=True).reset_index()
        dfFlattened = dat.melt(id_vars=['index', 'label'], value_vars=dat.columns.tolist())
        dfFlattened.columns = ['idx', 'label', 'patterns', 'value']
        dfFlattened = dfFlattened.sort_values(by=['label', 'patterns'])
        
        dx = dfFlattened[(dfFlattened['value']>0)]
    
        colvals = sns.color_palette("tab10")
        colsel = [colvals[classNo]]
        g = sns.relplot(x="idx", y="patterns", hue='label',
                        data=dx, s=15, palette=colsel, height=figsize[0]*2, aspect=1.5, legend=None)
                                #data=dx, s=25, palette=sns.color_palette("tab10"), height=figsize[0]*2, aspect=1.5)
        axes = g.axes.flat[0]
        px=axes.yaxis.set_tick_params(labelsize=9)
        px=axes.xaxis.set_tick_params(labelsize=9, rotation=90)
        px=axes.set_ylabel('')
        px=axes.set_xlabel('data instances')
        px=axes.get_xaxis().set_ticks([])

    def plot_relplot_byclass_threshold(self, clf, classNo=1, threshold=0.7, scale=0.5):
        random.seed(1729)
        a, b = clf.x_test_hup_.todense(), clf.y_pred_
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        df = df[df.label==classNo]
        
        figsize=(.1*len(df.columns)*scale, 0.2*df.shape[1]*scale) #row size, col size
        cvals = list(set(clf.y_pred_))
        datFinal = df
        prob = clf.y_pred_proba_.T[classNo]
        
        dfx = pd.concat([df, pd.DataFrame(prob, columns=['prob'])], axis=1)
        dfx1 = dfx.sort_values(by=['prob'], ascending=False)
        dfx1 = dfx1[dfx1['prob']>=threshold]
        dat = dfx1.iloc[:, :-1]#discard the last prob column
            
        datt = dat.T
        dat = datt[(datt.T != 0).any()].T #drop all zero cases
        if classNo==0:#zero label gets removed for 0 cases
            dat['label'] = [0]*dat.shape[0]
        dcntGrouped = dat
        dcntGrouped = dcntGrouped.reset_index()
        dcnt = dict((dat.sum(axis=0)))
    
        dat = dat[dat.columns].reset_index(drop=True).reset_index()

        assert dat.shape[0]>0, 'no instance match the required minimum threshold, reduce threshold and try again'
        
        dfFlattened = dat.melt(id_vars=['index', 'label'], value_vars=dat.columns.tolist())
        dfFlattened.columns = ['idx', 'label', 'patterns', 'value']
        dfFlattened = dfFlattened.sort_values(by=['label', 'patterns'])
        
        dx = dfFlattened[(dfFlattened['value']>0)]
    
        colvals = sns.color_palette("tab10")
        colsel = [colvals[classNo]]
        g = sns.relplot(x="idx", y="patterns", hue='label',
                        data=dx, s=15, palette=colsel, height=figsize[0]*2, aspect=1.5, legend=None)
                                #data=dx, s=25, palette=sns.color_palette("tab10"), height=figsize[0]*2, aspect=1.5)
        axes = g.axes.flat[0]
        px=axes.yaxis.set_tick_params(labelsize=9)
        px=axes.xaxis.set_tick_params(labelsize=9, rotation=90)
        px=axes.set_ylabel('')
        px=axes.set_xlabel('data instances')
        px=axes.get_xaxis().set_ticks([])
    
    def plot_relplot_2d(self, clf, y_test, threshold=-1, classNoSel=-1, maxrows=-1):
        random.seed(1729)
    
        c1, c2 = [], []
        idxesm = []
        classx = []
        dout = {}
    
        foldNo = 0  
        cvals = list(set(clf.y_pred_))
        for classNo in cvals:
            if classNoSel!=-1:
                if classNo!=classNoSel: continue 
            
            a, b = clf.x_test_hup_.todense(), clf.y_pred_
            df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
            df = df[df.label==classNo]
            
            cvals = list(set(clf.y_pred_))
            datFinal = df
            prob = clf.y_pred_proba_.T[classNo]
    
            idxes = np.where(prob>threshold)[0] #indexes where value is greater than threshold
            
            ytest = y_test
            a, b = clf.x_test_hup_.todense(), clf.y_pred_ #x_test_fold, y_pred
            df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
            dat = df.iloc[idxes, :-1]#exclude label column
    
            datt = dat.T
            dat = datt[(datt.T != 0).any()].T #drop all zero cases
            
            res = sorted(dat.columns.tolist())
            
            if maxrows==-1:
                maxrows = 10
            szCntr = 0
            for dxi, dx in dat.iterrows():
                if szCntr>=maxrows: break
                flag = False
                #print(dx)
                for ci in dx[dx>0].index.tolist():#non-zero entries
                    cx = str(ci).split(',')
                    if len(cx)>1:
                        flag=True
                        c1.append(cx[0]); c2.append(cx[1]); idxesm.append(dxi); classx.append(classNo)
                if flag: szCntr+=1
    
            print(classNo, 'size ', dat.shape[0], ' act size ', szCntr)#act size may be zero if all patterns are of size 1 only (even though huiMaxLen is set to a value>1)
    
        if szCntr==0:
            print('2d plot can\'t be generated when huiMaxLen (or L) is equal to 1. Set L > 1 to generate patterns of size greater than 1.')
            return 
    
        dout['c1'] = c1
        dout['c2'] = c2
        dout['idx'] = idxesm
        dout['class'] = classx #class column
        doutx = pd.DataFrame(dout)
    
        doutx = doutx.sort_values(by=['c2', 'c1'])
        sn=sns.relplot(x="c1", y="c2", hue='idx', style='class', data=doutx, s=50, palette=sns.color_palette("tab10"), aspect=2.5,
                       alpha=0.7)#, legend=None)
        #sn=sns.swarmplot(x="c1", y="c2", hue='idx', data=doutx, s=5, palette=sns.color_palette("tab10"))
        
        #jitter 
        axes = sn.axes.flat[0]
        dots = axes.collections[-1]
        offsets = dots.get_offsets()
        #print('offsets ', offsets)
        jittered_offsets = offsets + np.random.uniform(0, .2, offsets.shape)
        dots.set_offsets(jittered_offsets)
    
        plt.yticks(fontsize=16)
        pl=plt.xticks(rotation=90, fontsize=16)
        plt.gca().set_ylabel('')
        plt.gca().set_xlabel('')
    
    
    def patterns_processed(self, a):
        singleItems = []
        pairItems = {}
        pair3Items = {}
        for ai in a:
            li = ai.split(',')
            if len(li)==1: 
                singleItems.append(li[0])
            elif len(li)==2:
                if not li[0] in pairItems.keys():
                    pairItems[li[0]] = [li[1]]
                else: 
                    pairItems[li[0]].append(li[1])
            elif len(li)==3:
                if not li[0] in pair3Items.keys():
                    pair3Items[li[0]] = [li[1:]]
                else: 
                    pair3Items[li[0]].append(li[1:])
            else: continue 
    
        print('original size ', len(a))
        cs = getCondensedPatterns_short(singleItems)
        print('single items (condensed) ', cs)
    
        if len(pairItems.keys())>0:
            print('pair items ')
            for k, v in pairItems.items():
                print(k, v)
            print(len(pairItems.keys()))
    
        if len(pair3Items.keys())>0:
            print('pair3 items ')
            for k, v in pair3Items.items():
                print(k, v)
            print(len(pair3Items.keys()))
    
    def get_patterns_byclass(self, clf, y_test, classNo=1, threshold=-1):
        random.seed(1729)
        foldNo = 0
        a, b = clf.x_test_hup_.todense(), clf.y_pred_
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        df = df[df.label==classNo]
        
        cvals = list(set(clf.y_pred_))
        datFinal = df
        prob = clf.y_pred_proba_.T[classNo]
    
        idxes = np.where(prob>threshold)[0] #indexes where value is greater than threshold
        
        ytest = y_test
        a, b = clf.x_test_hup_.todense(), clf.y_pred_ #x_test_fold, y_pred
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        dat = df.iloc[idxes, :-1]#exclude label column
    
        datt = dat.T
        dat = datt[(datt.T != 0).any()].T #drop all zero cases
        
        res = sorted(dat.columns.tolist())
        self.patterns_processed(res) #condense size 1
    
    def getCondensedPatterns_short(self, pset):
        px = sorted(pset)
        psetDict = {}
        import re
        for lidx, l in enumerate(px):
            lx = l.split(',')
            #print(lx)
            prefix = ':'.join([lxi.split('=')[0] for lxi in lx])
            val = [1 if len(lxi.split('='))==1 else lxi.split('=')[1] for lxi in lx]
            val = [vali.replace('[', '').replace(']', '') if vali!=1 else vali for vali in val]
            #print(vx)
            if psetDict.get(prefix, -1)==-1: #not found
                psetDict[prefix] = []
            
            psetDict[prefix].extend(val) #feature rank, value
    
        print('original (condensed) size ', len(pset), len(psetDict.keys()))
        return psetDict
    
    def plot_map_instance(self, clf, idx=0, full=False): #test data, random instance
        random.seed(1729)
        
        a,  b = clf.x_test_hup_.todense(), clf.y_pred_
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        dat = df.iloc[:, :-1]#exclude label column
        
        lbl = b[idx]#label column
        print('lbl ', lbl)
        dat.columns = clf.procdata_['patternsMapped']
        inst = dat.iloc[idx, :].to_frame()
        inst.columns = ['pattern']
        if full: display('full', inst)
        inst = inst.loc[(inst.pattern==1)].T
        display(inst)
    
    def plot_map_instance_random(self, clf, y_test, threshold=0, classNo=1, full=False): #only on test data using y_pred_proba, filter instance above prediction threshold, and select randomly
        prob = clf.y_pred_proba_.T[classNo] #y_pred_proba on test
        
        idxes = np.where(prob>threshold)[0] #indexes where value is greater than threshold
        #np.random.seed(30)
        idx = idxes[np.random.choice(len(idxes))]
        print('index selected ', idx, 'prob value ', prob[idx])
    
        ytest = y_test
        a,  b = clf.x_test_hup_.todense(), clf.y_pred_ #x_test_fold, y_pred
        #print('b vals ', b)
        df = pd.concat([pd.DataFrame(a, columns=clf.procdata_['patternsMapped']), pd.DataFrame(b, columns=['label'])], axis=1)
        dat = df.iloc[:, :-1]#exclude label column
        lbl = b[idx]#label column
        print('predicted, actual label ', lbl, ytest[idx])
        dat.columns = clf.procdata_['patternsMapped']
        inst = dat.iloc[idx, :].to_frame()
        inst.columns = ['pattern']
        if full: display('full', inst)
        inst = inst.loc[(inst.pattern==1)].T
        display(inst)
    
    def get_pred_proba_plots(self, clf):
        plt.figure(figsize=(10,3))
        plt.subplot(1, 2, 1)
        px=sns.histplot(clf.y_pred_proba_[:, 0])
        px=plt.title('class 0')
        plt.subplot(1, 2, 2)
        px=sns.histplot(clf.y_pred_proba_[:, 1]) 
        px=plt.title('class 1')
    
    def get_lr_feature_importance(self, clf, topN=10):
        assert isinstance(clf.base_estimator, LogisticRegression), 'base estimator should be Logistic Regression '
        
        fi = pd.DataFrame({'HUG pattern' : clf.procdata_['patternsMapped'], 'Coefficient': clf.base_estimator.coef_[0].tolist()})
        fi['Importance'] = np.abs(fi['Coefficient'])
        fiScores = fi.sort_values(by=['Importance'], ascending=False).iloc[:topN]
    
        display(HTML(fiScores.to_html(index=False)))
        fiScores.plot(x='HUG pattern', y='Importance', kind='barh', figsize=(5, 3), color='orange')
    
    def get_permutation_feature_importance(self, model, y_test, topN=10):
        xtmp = model.x_test_hup_.toarray()
        out = permutation_importance(model.base_estimator, xtmp, y_test, n_repeats=10, random_state=0, n_jobs=2)
        sorted_idx = out.importances_mean.argsort()[::-1][:topN]
    
        fig, ax = plt.subplots()
        plt.figure(figsize=(5,3))
        ax.boxplot(out.importances[sorted_idx].T, vert=False, labels=np.array(model.procdata_['patternsMapped'])[sorted_idx])
        ax.set_title("Permutation Feature Importance (test data)")
        fig.tight_layout()
        plt.show()
    
    def get_permutation_feature_importance_train(self, model, y_train, topN=10):
        xtmp = model.procdata_['x_train_hup'].toarray()
        out = permutation_importance(model.base_estimator, xtmp, y_train, n_repeats=10, random_state=0, n_jobs=2)
        sorted_idx = out.importances_mean.argsort()[::-1][:topN]
    
        fig, ax = plt.subplots()
        plt.figure(figsize=(5,3))
        ax.boxplot(out.importances[sorted_idx].T, vert=False, labels=np.array(model.procdata_['patternsMapped'])[sorted_idx])
        ax.set_title("Permutation Feature Importance (train data)")
        fig.tight_layout()
        plt.show()
        