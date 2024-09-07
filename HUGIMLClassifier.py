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

import pandas as pd, numpy as np, datatable as dt, copy, math, time, subprocess, os, glob
from itertools import combinations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler, LabelBinarizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scipy import io
from scipy.stats import entropy
from scipy.sparse import csr_matrix
import struct

import warnings
warnings.filterwarnings("ignore")

class HUGIMLClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, 
                 allCols=None, origColumns=None, 
                 B=5, L=1, G=1e-4, 
                 dsName="unspecifiedClf", 
                 foldNo=1, 
                 base_estimator=None, 
                 imbWeights=1, 
                 huiItemsPercent=1, topK=-1, fsK=-1,
                 verbose=False):
        
        self.allCols, self.origColumns = allCols, origColumns 
            #mandatory, should be set based on actual data
            #allCols has column names of integers, floats, and categorical columns in 3 lists e.g. format [[], [], []]
            #origColumns - list of column names
        self.B, self.L, self.G = B, L, G #optional - key parameters to be tuned for improving model performance
            #number of bins (B) is auto determined based on data; L or max HUI length is set to 1 by default; G is set to 1e-4 by default
        
        self.dsName = dsName #optional, for labeling verbose output
        self.foldNo = foldNo #optional, default 1, used for labeling verbose output and generating intermediate file names
        self.base_estimator = base_estimator #optional, default logistic regression 
        self.imbWeights = imbWeights #imbWeights, default 1, set it to higher value to give more weights to minority class
        self.huiItemsPercent, self.topK, self.fsK = huiItemsPercent, topK, fsK #optional for improving computational performance/generating simpler models
        self.verbose = verbose #optional, default False

    def get_column_indices(self, origColumns, column_groups):
        numInt, numIntFloat = len(self.allCols[0]), len(self.allCols[0])+len(self.allCols[1])
        numericIntCols_indices = [idx for idx in range(numInt)]
        numericFloatCols_indices = [idx for idx in range(numInt, numIntFloat)]
        catCols_indices = [idx for idx in range(numIntFloat, len(self.origColumns))]
        
        return [numericIntCols_indices, numericFloatCols_indices, catCols_indices]

    def write_cat_columns_to_binary(self, fname, string_array):
        with open(fname, 'wb') as f:
            for row in string_array:
                row_data = ','.join([str(r) for r in row])
                row_bytes = row_data.encode('utf-8')
                f.write(len(row_bytes).to_bytes(4, byteorder='little'))#write the length of the row
                f.write(row_bytes)#write the row data

    def read_sparse_matrix_binary(self, filename):
        with open(filename, 'rb') as fx:
            rowsCnt = int.from_bytes(fx.read(4), 'big')
            colsCnt = int.from_bytes(fx.read(4), 'big')
            nnz = int.from_bytes(fx.read(4), 'big')
            #print(rowsCnt, colsCnt, nnz)
            
            row_indices = np.zeros(nnz, dtype=int)
            col_indices = np.zeros(nnz, dtype=int)
            data = np.ones(nnz, dtype=int)
    
            for i in range(nnz):
                row_indices[i] = int.from_bytes(fx.read(4), 'big')
                col_indices[i] = int.from_bytes(fx.read(4), 'big')
                
        sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(rowsCnt, colsCnt))
        return sparse_matrix

    def hupTransform_test_new(self, X_test):#apply HUI patterns on test data
        base_filename = 'outputs/inpdata/'+self.dsName
        if len(self.allColsIdx[0])>0: X_test[:, self.allColsIdx[0]].T.astype(np.int32).tofile(f'{base_filename}_x_test_int.bin')
        if len(self.allColsIdx[1])>0: X_test[:, self.allColsIdx[1]].T.astype(np.float32).tofile(f'{base_filename}_x_test_float.bin')
        if len(self.allColsIdx[2])>0: self.write_cat_columns_to_binary(f'{base_filename}_x_test_cat.bin', X_test[:, self.allColsIdx[2]].T)
        
        outputPath = 'outputs/hui/'    
        runCmd = 'java -Xms1g -Xmx6g -jar THUIsl.jar'
        runCmd += ' dsname='+self.dsName+' foldno='+str(self.foldNo)+' modeltest=true'
        runCmd += ' numRows='+str(X_test.shape[0])+" numIntCols="+str(len(self.allCols[0]))
        runCmd += " numFloatCols="+str(len(self.allCols[1]))+" numCatCols="+str(len(self.allCols[2]))
        #print('runCmd ', runCmd)
        out = subprocess.call(runCmd, shell=True, stdout = subprocess.PIPE)
        
        #outputFileSparseTid = self.dsName + '_tid_sparse_test_'+str(self.foldNo)+'.bin'
        outputFileSparseTid = self.dsName + '_tid_sparse_test.bin'
        x_test_hup = self.read_sparse_matrix_binary(outputPath+outputFileSparseTid) #sparse matrix
        
        if self.verbose: print('X test hup size ', x_test_hup.shape)
        return x_test_hup
        
    def runTopKhui_train_new(self, numRows):
        topK = self.topK
        fsK = self.fsK

        minutility = 1e10
        outputPath = 'outputs/hui/'
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)

        runCmd = 'java -Xms1g -Xmx6g -jar THUIsl.jar'
        runCmd += ' topK='+str(topK)+' fsK='+str(fsK)+' B='+str(self.B)+' L='+str(self.L)+' G='+str(self.G)+' dsname='+self.dsName+' foldno='+str(self.foldNo);
        runCmd += ' numRows='+str(numRows)+" numIntCols="+str(len(self.allCols[0]))+" numFloatCols="+str(len(self.allCols[1]))+" numCatCols="+str(len(self.allCols[2]))
        runCmd += ' numClasses='+str(len(self.classes_));
        #print('runCmd ', runCmd)
        out = subprocess.call(runCmd, shell=True, stdout = subprocess.PIPE)
        
        #outputFileUtil = self.dsName + '_util_fs_mapped_'+str(self.foldNo)+'.txt'
        outputFileUtil = self.dsName + '_util_fs_mapped.txt'
        df = dt.fread(outputPath+outputFileUtil, sep=' ', quotechar="\'")
        assert df.shape[0]!=0, 'no HUI patterns found, change parameters and re-run'
        
        #df.names = ['utility', 'pattern', 'ig', 'entropy', 'pureClass']
        patternsMapped = df['pattern'].to_list()[0]
        
        #read train data
        #outputFileSparseTid = self.dsName + '_tid_sparse_'+str(self.foldNo)+'.bin'
        outputFileSparseTid = self.dsName + '_tid_sparse.bin'
        x_train_hup = self.read_sparse_matrix_binary(outputPath+outputFileSparseTid)  #txnPatternMatrix - |T| x |P| (o-1 matrix)
        
        return x_train_hup, patternsMapped

    
    def hupTransform_train_new(self, X_train, y_train):   
        self.allColsIdx = self.get_column_indices(self.origColumns, self.allCols)#pick indices of columns of different types

        procdata = {}
        base_filename = 'outputs/inpdata/'+self.dsName
        if len(self.allColsIdx[0])>0: X_train[:, self.allColsIdx[0]].T.astype(np.int32).tofile(f'{base_filename}_x_train_int.bin')    
        if len(self.allColsIdx[1])>0: X_train[:, self.allColsIdx[1]].T.astype(np.float32).tofile(f'{base_filename}_x_train_float.bin')
        if len(self.allColsIdx[2])>0: self.write_cat_columns_to_binary(f'{base_filename}_x_train_cat.bin', X_train[:, self.allColsIdx[2]].T)
        y_train.T.astype(np.int32).tofile(f'{base_filename}_y_train.bin')
        #colQuoted = np.array([f'{i},"{col}"' for i, col in enumerate(self.origColumns)], dtype=str)
        #np.savetxt(f'{base_filename}_allColsIdxToName.txt', colQuoted, fmt='%s')
        colQuoted = np.array([f'"{col}"' for col in self.origColumns], dtype=str).reshape(-1, 1)
        #print(colQuoted1.T)
        self.write_cat_columns_to_binary(f'{base_filename}_allColsIdxToName.bin', colQuoted.T)
        
        if self.verbose: print("starting topk hui ", time.strftime("%H:%M:%S", time.localtime()))
    
        #set or update topk, fsk parameters
        huiItemsPercent = self.huiItemsPercent #default is 100%
        nitems = 100#len(procdata['relItems'].items())
        
        lsize = {}
        for i in range(1, 7): lsize[i] = 0
        for i in range(1, 7):
            lsize[i] += math.comb(nitems, i) #cumulative value
        
        updated=False
        if self.L==-1 or self.L==1:#unspecified case, huiMaxLen defaults to 1 
            newTopK = lsize[1]
            updated=True
        elif self.L!='all' and self.L>=2 and self.L<=6:
            newTopK = huiItemsPercent*lsize[self.L]#1 and 2 itemsets
            updated=True
        else:#huiMaxLen set to 'all' or any other value
            if self.topK==-1:#topK not set for 'all' or other huiMaxLen cases, set it to fraction of size for huiMaxLen=2
                newTopK = huiItemsPercent*lsize[2]
                updated = True
            else: updated=False
            
        if updated:
            if self.topK==-1:#topK not set
                procdata['topKoriginal'] = self.topK
                self.topK = int(newTopK)
            
            if self.fsK==-1 or self.fsK>self.topK:
                procdata['fsKoriginal'] = self.fsK
                self.fsK = self.topK
            
        x_train_hup, patternsMapped = self.runTopKhui_train_new(X_train.shape[0])
        if self.verbose: print(self.topK, " actual number of itemsets generated ", len(patterns))

        procdata['patternsMapped'] = patternsMapped 
        procdata['x_train_hup'] = x_train_hup
        if self.verbose: print('X train hup size ', x_train_hup.shape)
        
        return procdata

    def get_bins(self):
        check_is_fitted(self) #sklearn validation
        filename = 'outputs/feModels/'+self.dsName+'_kbins.bin'
        intFloatCols = len(self.allCols[0])+len(self.allCols[1])
        numBins = []
        try:
            with open(filename, 'rb') as fx:    
                for i in range(intFloatCols):
                    nb = int.from_bytes(fx.read(4), 'big')
                    fx.read(4*(nb+1))#skip reading edges; use #struct.unpack('>f', fx.read(4))[0] to read edges
                    numBins.append(nb)
        except IOError as e:
            print(f"error reading file: {e}")
        return numBins
        
    def get_hug_features(self):
        check_is_fitted(self) #sklearn validation
        return self.procdata_['patternsMapped']

    def get_transformed_shape(self):
        check_is_fitted(self) #sklearn validation
        return self.procdata_['x_train_hup'].shape
    
    def cleanupFolderFiles(self):#delete existing processed files
        if self.foldNo==1:#when invoking first fold, remove all the old files in the directory
            if os.path.exists('outputs'):
                time.sleep(.5)#extra sleep time added to avoid process sync issues (during grid search or cross validation with multiple quick runs)
                for f in glob.glob('outputs/'+self.dsName+'*.txt'): os.remove(f)
                for f in glob.glob('outputs/feModels/'+self.dsName+'*.txt'): os.remove(f)
                for f in glob.glob('outputs/inpdata/'+self.dsName+'*.txt'): os.remove(f)
                for f in glob.glob('outputs/inpdata/'+self.dsName+'*.bin'): os.remove(f)
                for f in glob.glob('outputs/hui/'+self.dsName+'*.txt'): os.remove(f)
                for f in glob.glob('outputs/hui/'+self.dsName+'*.bin'): os.remove(f)
    
            if not os.path.exists('outputs/'):
                os.makedirs('outputs/')
            inputPath = 'outputs/inpdata/'
            if not os.path.exists(inputPath):
                os.makedirs(inputPath)
            if not os.path.exists('outputs/feModels/'):
                os.makedirs('outputs/feModels/')

    def validateParams(self):
        assert (self.allCols!=None and self.allCols!='') and (self.origColumns!=None and self.origColumns!=''), 'Specify mandatory arguments:: \n\t\t allCols: [list of integer, list of float, list of categorical columns], \n\t\t origColumns: original column names of the data frame \n\t\t If you are using your own dataset, call prepareXy(X, y) on the original data before calling fit and train/test split'
        if self.G==0: self.G=0.0 #use float value
        assert isinstance(self.B, int) and isinstance(self.L, int) and isinstance(self.G, float), 'give the correct type of arguments: B and L should be integers, and G should be a float value'
        
        if self.dsName=='' or (not isinstance(self.dsName, str)):#dsName is used as an argument in topKHUI function. can't be left blank for correct argument parsing 
            self.dsName = 'unspecified'

    def prepareXy(self, X, y):#required for correct processing of data in HUGIML
        assert type(X)==pd.core.frame.DataFrame, 'X should be a pandas data frame '
            
        X.columns = [str(c) for c in X.columns.tolist()]
        numericColumns = [colx for colx in X.columns.tolist() if not np.issubdtype(X[colx].dtype, object)]
        catColumns = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]

        numericIntCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, np.integer)]
        numericFloatCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, float)]
        catCols = [colx for colx in X.columns.tolist() if np.issubdtype(X[colx].dtype, object)]
        
        allCols = [numericIntCols, numericFloatCols, catColumns]
        X = X[[ai for a in allCols for ai in a]]
        X = X.reset_index(drop=True)
        self.allCols = allCols
        self.origColumns = X.columns.tolist()

        assert type(y)==pd.core.series.Series, 'y should be a pandas series object '
        u, cnts = np.unique(y, return_counts=True)
        rx = sorted([(a, b) for a, b in zip(u, cnts)], key=lambda x: -x[1])
        mapYbyCntsDesc = dict([(rxi[0], ridx) for ridx, rxi in enumerate(rx)])
        y = pd.Series(y).map(mapYbyCntsDesc).to_numpy() #transformed y - higher value assigned to class with smaller number of examples
        self.yNewToOriginal = mapYbyCntsDesc #maintain mapping of transformation
        
        return X, y    
    
    def fit(self, X_train, y_train):
        if type(X_train)==pd.core.frame.DataFrame:
            X_train = X_train.to_numpy()

        if  type(y_train)==pd.core.series.Series:
            y_train = y_train.to_numpy()
        
        assert type(X_train)==np.ndarray, 'X_train should be a numpy array '
        assert type(y_train)==np.ndarray, 'y_train should be a numpy array '
        
        X_train, y_train = check_X_y(X_train, y_train, dtype=None) #sklearn validation 
        
        if self.base_estimator==None:
            self.base_estimator = LogisticRegression(solver='liblinear', random_state=0, max_iter=500)
        self.model = Pipeline([('clf', self.base_estimator)])
            
        self.n_features_in_ = X_train.shape[1]
        self.classes_ = np.unique(y_train)

        self.validateParams()
        self.cleanupFolderFiles()

        if self.verbose: print('\nHUGIML Classifier ')
        self.x_test_hup_ = None #set it to None, before fitting data again
        self.y_pred_proba_, self.y_pred_ = None, None

        #transform data
        self.procdata_ = self.hupTransform_train_new(X_train, y_train)
        if self.verbose: print('transformed train shape ', self.procdata_['x_train_hup'].shape)
        
        #fit model
        self.model.fit(self.procdata_['x_train_hup'], y_train)
        return self

    def predict_proba(self, X_test):
        if type(X_test)==pd.core.frame.DataFrame:
            X_test = X_test.to_numpy()
            
        check_is_fitted(self) #sklearn validation
        X_test = check_array(X_test, dtype=None) #sklearn validation

        if self.x_test_hup_==None:
            self.x_test_hup_ = self.hupTransform_test_new(X_test)
        
        self.y_pred_proba_ = self.model.predict_proba(self.x_test_hup_)
        self.y_pred_ = np.argmax(self.y_pred_proba_, axis=1)#required for subsequent plot generation (or explicitly call x_test hup transform again)
        return self.y_pred_proba_

    def predict(self, X_test):
        if type(X_test)==pd.core.frame.DataFrame:
            X_test = X_test.to_numpy()
            
        check_is_fitted(self) #sklearn validation
        X_test = check_array(X_test, dtype=None) #sklearn validation

        if self.x_test_hup_==None:#once fitted, re-use the transformations
            self.x_test_hup_ = self.hupTransform_test_new(X_test)
        return self.model.predict(self.x_test_hup_)
