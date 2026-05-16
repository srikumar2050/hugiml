# HUG-IML: High Utility Gain Patterns for Interpretable Machine Learning

The repository provides the source code of a classifier modeling method (HUG-IML). High Utility Gain-Interpretable Machine Learning (HUG-IML) is an intrinsic classifier model that extracts a class of higher order patterns and embeds them into an interpretable learning model such as logistic regression. The model supports both binary and multi-class classification problems. The specific details of the HUG-IML models, benchmark results, and their applications can be referred to in the IEEE Access paper titled: Interpretable classifier models for decision support using high utility gain patterns, IEEE Access 2024, DOI: https://doi.org/10.1109/ACCESS.2024.3455563.

If you use the software programs in this repository, please cite the following paper:

```
    @article{krishnamoorthy2024,
              title={Interpretable classifier models for decision support using high utility gain patterns},
              author={Krishnamoorthy, Srikumar},
              journal={IEEE Access},
              year={2024},
              doi={https://doi.org/10.1109/ACCESS.2024.3455563}
    }
```

<br/>

### 1. REPOSITORY INFORMATION

This repository primarily contains sklearn (scikit-learn.org) compatible python source files and java programs. The python related files are used for reading the dataset and invoking standard APIs for classifier modeling, hyper-parameter tuning, and performance evaluation. The core data transformations and pattern mining are performed using java programs. The java programs are invoked directly from the python program (HUGIMLClassifier.py) and relevant data are exchanged between programs using binary and text files. A separate output folder is created to store the generated files. 

The overall classifier modeling process workflow is shown in Figure 1. 

<figure>
  <img alt="HUG-IML Process Workflow" src="HUG-IML Process Workflow.png" width="780" height="540"/>
  <figcaption><strong>Figure 1.</strong> HUG-IML Process Workflow</figcaption>
</figure>
<br/>
The specific details of program/data files and directories contained in this repository are as follows:

#### 1.1 Benchmark datasets

datasets: This directory contains four benchmark binary classifier modeling datasets:

1. Portugese Bank Telemarketing: UCI (https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. Home Equity Line Of Credit (HELOC): FICO (https://community.fico.com/s/explainable-machine-learning-challenge)
3. Pima Indian Diabetes: National Institute of Diabetes and Digestive and Kidney Diseases (http://archive.ics.uci.edu/ml, https://data.world/uci/pima-indians-diabetes)
4. Titanic: Open ML (https://www.openml.org/search?type=data&sort=runs&id=40945&status=active)

The utils.py file refers to this directory for reading the benchmark datasets.

#### 1.2 Python related files

Main files

1. HUGIML Classifier Sample Notebook.ipynb: Use this notebook to run the HUG-IML classifier, perform hyper-parameter tuning, and evaluate classifier modeling performance.
2. HUGIMLClassifier.py: This file provides the core sklearn (scikit-learn.org) compatible APIs for classifier modeling. It invokes the THUIsl.jar to perform data transformations and utility pattern mining.
3. utils.py: Helper files are provided to read the dataset, compute performance metrics, and generate visualization of the classifier model results.

Setup/configuration files

1. requirements.txt: This file provides all the dependent python libraries needed for running the program.
2. pythonVirtualEnvironmentSetup.txt: This file provides step-by-step instructions for setting up a virtual environment in python and avoid any possible package level conflicts. The requirements.txt file referred in the installation step (step #5) is used to install all the dependent libraries.

#### 1.3 Java related files

Binary files

THUIsl.jar: Compiled java files. Java(TM) SE Runtime Environment (build 22.0.1+8-16) was used for compiling the java files.

Source files

THUIsl_src: This directory contains the source code of the java files used for data transformation and utility pattern mining. Use these files if you wish to manually compile the program on your JVM or customize it as per your needs. The specific source files included are:

    * RunTHUISlPrep.java          The class that is invoked from the python program by passing necessary parameters
    * AlgoTHUIsl.java             The core class that perform HUG pattern mining
    * TransactionGenerator.java   Generate transaction level data from the input training/test data
    * Pattern.java                Maintain the mined HUG patterns
    * UtilityList.java            Program to store utility information of individual items
    * Element.java                Program that stores the individual transaction level details of items
    * Item.java                   A helper class
    * ItemUtility.java            A helper class to maintain item and its utility values
    * MinMaxScaler.java           Perform min-max scaling transformations
    * KBinsDiscretizer.java       Discretize the numerical variables based on user-specified or estimated bins
    * LabelBinarizer.java         Program to encode categorical data
    * NMI.java                    Program to compute Normalized Mutual Information
    * CorrelationCustom.java      Pearson correlation computations
    * META-INF/MANIFEST.MF        Specifies the main class used by the jar file

Top-K High Utility Itemset (THUI) is a Top-K high utility pattern mining algorithm introduced in the article: Krishnamoorthy, S. (2019). Mining top-k high utility itemsets with effective threshold raising strategies.  *Expert Systems with Applications* ,  *117* , 148-165. The core THUI algorithm was built using the SPMF code base (https://www.philippe-fournier-viger.com/spmf/) distributed under the GPL v3 license. THUIsl extends the basic version of THUI algorithm and uses supervised label information for pattern mining. The algorithm performs data transformations (scaling, discretization, and categorical encoding), transaction generation with utility information based on supervised labels, and mines a new class of High Utility Gain (HUG) patterns that aids the downstream learning task. It also transforms the training and test data based on the extracted HUG patterns for interpretable machine learning.

#### 1.4 Application containers

Dockerfile: This file includes necessary steps for installing the relevant python and java libraries. Refer to the Usage Information (Option II) below for the actual deployment steps.

#### 1.5 Data related files

outputs directory: This directory is not part of this shared repository. It is automatically created during the actual execution of the program. It will have 3 sub-folders named inpdata, feModels, and hui. The following files are created in each of these folders during the program execution. The files created in this 'outputs' folder are automatically deleted by the python program before every classifier fit operation.

* `<datasetName>`_colNamesNew.bin
  Contains the column names generated by the Java program after data transformations. The file is generated during training stage and subsequently used at the testing stage.

a. outputs/inpdata directory:This directory stores the input data required for data transformation and utility pattern mining.

    * `<datasetName>`_x_{train|test}_{int|float|cat}.bin The train and test binary data created during the initial python program execution for subsequent use by Java program for pattern mining.
      A maximum of 6 files are created based on the nature of input data (i.e. based on whether the dataset has integer, float, and categorical columns)
    * `<datasetName>`_y_train.bin The label (or outcome) information created during the initial python program execution at the training stage.
    * `<datasetName>`_allColsIdxToName.bin
      The column name information generated during the initial python program execution at the training stage.

b. outputs/feModels directory:This directory stores the transformation parameters generated by the HUG-IML model. It uses the files created in the outputs/inpdata folder while performing transformations.

    * `<dsName>`_ms.bin Min-max scaler data transformation parameters generated during the training stage by the Java program.
    * `<dsName>`_kbins.bin K-bins discretizer data transformation parameters generated during the training stage by the Java program.
    * `<dsName>`_lb.bin
      Label binarizer data transformation parameters generated during the training stage by the Java program.
      The above generated files are then read during the testing stage to extract the learnt parameters and apply them on the test data.

c. outputs/hui directory: This directory contains the mined utility patterns and the HUG transformed data matrix.

    * `<dsName>`_util_fs.bin This file contains the mined utility patterns.
    * `<dsname>`_util_fs_mapped.txt This is a human readable text file that contains the same set of patterns in the `<dsName>`_util_fs.bin file. While the `<dsName>`_util_fs.bin file contains internally generated item identifiers, the `<dsName>`_util_fs_mapped.txt file maps the internal identifiers to the actual column names and discretized bins/categorical column values.
    * `<dsName>`_tid_sparse.bin This is the HUG-transformed training data file. It primarily maps the original training data to the mined pattern space. The dimensionality of this data matrix (in dense form) will be |size of training data| x |number of mined patterns|. The actual data is stored and processed in sparse matrix format. This binary file is read at the python program end for fitting the final interpretable classifier model (e.g. logistic regression) on the transformed data.
    * `<dsName>`_tid_sparse_test.bin
      This is the HUG-transformed test data file. It uses the mined patterns generated during the training stage (stored in `<dsName>`_util_fs.bin) and applies it on the test data. The dimensionality of this matrix (in dense form) will be |size of test data| x |number of mined patterns|. This binary file is read at the python program end for final prediction using the fitted model.

The shared programs were tested on python version 3.9.18 and java jdk-22.0.1. In order to make the shared programs run successfully on your local machine, you may have to use the manual compilation and configuration steps provided in Usage Information below.

#### 1.6 License information

GNU GPLv3 License: This repository contains a free software program. You can redistribute it and/or modify it under the terms of GNU General Public Licence. The license details are shared in this file. It can also be referred to online at http://www.gnu.org/licenses/.

<br/>

### 2. USAGE GUIDELINES

You may use one of the following options for executing the HUG-IML classifier modeling programs shared in this repository.

<em>Option I: Run without any installation or configuration directly in Code ocean</em>
1. Use the URL: https://doi.org/10.24433/CO.0007155.v1
2. Run the python notebook file (HUGIML Classifier Sample Notebook.ipynb) 

<em>Option II: Application containers</em>

1. Open the Docker Desktop Application on your machine
2. Open the Dockerfile shared in Visual Studio (or other related IDEs)
3. Build the image
4. Deploy and run the container
5. The application runs on http://localhost:3333/lab
6. Open the HUGIML Classifier Sample Notebook.ipynb file to run the HUG-IML classifier, perform hyper-parameter tuning, and evaluate classifier modeling performance.

<em>Option III: Basic python and java setup</em>

1. Install the dependent python libraries provided in the requirements.txt file.
   e.g. pip install -r requirements.txt
2. THUIsl.jar shared in the repository is prepared by compiling java files using Java(TM) SE Runtime Environment (build 22.0.1+8-16). If your JDK/JRE is incompatible with this version, then you may have to compile the java files (refer to THUIsl_src directory for the java files). Compile the java program and create jar file using the following steps:
   
   a. javac *.java
   
   b. jar cvfm THUIsl.jar META-INF/MANIFEST.MF *.class
3. Open the HUGIML Classifier Sample Notebook.ipynb and run the HUGIML classifier.

<em>Option IV: Manual compilation and configuration</em>

1. Use pythonVirtualEnvironmentSetup.txt to setup a virtual environment in python and install necessary dependencies. This can help avoid any package level conflicts you may encounter in your base python environment.
2. Use the java files in THUIsl_src folder, compile them and create a THUIsl.jar file. You may use the following steps for compiling and creating a jar file.
   
   a. javac *.java
   
   b. jar cvfm THUIsl.jar META-INF/MANIFEST.MF *.class
3. Place the compiled .jar file in the location where the relevant python files are stored (HUGIML Classifier Sample Notebook.ipynb, HUGIMLClassifier.py, and utils.py)
5. Open the HUGIML Classifier Sample Notebook.ipynb notebook and run the HUGIML classifier.

<br/>

### 3. BASIC MODELING STEPS AND PARAMETERS

The key modeling steps are provided below for illustration. You may refer to the sample notebook shared in the repository for detailed set of steps.

```
#import necessary files
from utils import DataUtils, MetricUtils, PlotUtils
from HUGIMLClassifier import HUGIMLClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
```

```
#set parameters
params = {
  'dsName': 'pimaIndianDiabetes', #label used as prefix for intermediate files, defaults to unspecifiedClf if unspecified
  'B': 7, #bin size to be used, computed if unspecified
  'L': 1, #length of HUG patterns, defaults to 1 if unspecified
  'G': 5e-3, #gain threshold, defaults to 1e-4 if unspecified
} 
#for other advanced (optional) parameters, refer to descriptions given in the constructor of the HUGIMLClassifier.py file.
```

```
#read the dataset and set mandatory parameters (allCols, origColumns)
X, y, yNewToOriginal, procdata  = DataUtils().get_dataset_df(params)
```

```
#initialize the classifier model
params = {**params, **procdata} #parameter is reset to include two mandatory fields allCols, origColumns
clf = HUGIMLClassifier(**params)
```

```
#train, test validation
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
```

```
#transform x by generating HUIs and fit a model
clf.fit(x_train, y_train)
```

```
#predict probability on test instances
y_pred_proba = clf.predict_proba(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)
```

```
#compute metrics
finalRes = MetricUtils().get_metrics(y_test, y_pred, y_pred_proba)
```

```
#display output
out = pd.DataFrame(finalRes).T
out.columns =['accuracy', 'f1', 'auc', 'hmeasure', 'logLoss', 'precision', 'recall']
out.index = ['performance']
display(out)
```

For detailed set of steps for using alternate base estimators, hyper-parameter tuning (grid search or optuna), performance evaluation, and visualization refer to the shared jupyter notebook file: HUGIML Classifier Sample Notebook.ipynb.

<br/>

### 4. PURE-PYTHON BACKEND (HUGIMLClassifierPy)

`HUGIMLClassifierPy` is a pure-Python re-implementation of the HUG-IML classifier that requires no Java runtime and no intermediate disk files. The public API is compatible with `HUGIMLClassifier`.

#### Installation

Install the required Python packages (no Java needed):

```
pip install numpy pandas scipy scikit-learn
```

#### Command-line usage

Run the sample script directly from the repository root:

```
python HUGIMLClassifierSample.py
```

Or use the provided sample notebook for an interactive walkthrough:

```
jupyter notebook "HUGIML Classifier Sample Notebook (Pure Python).ipynb"
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `B` | int | `5` | Quantile bins per numerical column. Use `-1` for supervised auto-selection (maximises per-column information gain over [2, 20]). |
| `L` | int | `1` | Maximum pattern length. `1` = singletons; `2` = singletons and pairs. |
| `G` | float | `1e-4` | Minimum information-gain threshold. Patterns below this are discarded. |
| `topK` | int | `-1` | Maximum patterns to retain. `-1` auto-computes as C(100, L). |
| `allCols` | list | `None` | `[int_cols, float_cols, cat_cols]` — column names grouped by type. Pair with `origColumns`. |
| `origColumns` | list | `None` | Ordered list of all column names matching the columns of X. Pair with `allCols`. |
| `base_estimator` | estimator | `None` | Downstream sklearn classifier. Defaults to `LogisticRegression`. |
| `verbose` | bool | `False` | Print progress during fit. |

#### Quick start

```python
from HUGIMLClassifierPy import HUGIMLClassifierPy
from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd

# Load data
data = pd.read_csv('datasets/pima indians diabetes.csv', header=None)
data.columns = ['numPregnancies','glucose','bp','skinThickness',
                'insulin','bmi','diabetesPedigreeFunction','age','class']
X, y = data.iloc[:,:-1], data.iloc[:,-1]

# Identify column types
numericIntCols   = [c for c in X.columns if np.issubdtype(X[c].dtype, np.integer)]
numericFloatCols = [c for c in X.columns if np.issubdtype(X[c].dtype, float)]
catCols          = [c for c in X.columns if np.issubdtype(X[c].dtype, object)]

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     random_state=0, stratify=y)

# Fit
clf = HUGIMLClassifierPy(B=-1, L=1, G=1e-6,
                         allCols=[numericIntCols, numericFloatCols, catCols],
                         origColumns=X.columns.tolist())
clf.fit(x_train, y_train)

# Predict
y_pred = np.argmax(clf.predict_proba(x_test), axis=1)

# Inspect patterns
print(clf.get_hug_features())   # human-readable pattern labels
print(clf.get_pattern_info())   # utility / information gain / support per pattern
```

#### Additional methods (available only in HUGIMLClassifierPy)

| Method | Returns | Description |
|---|---|---|
| `transform(X)` | `csr_matrix (n, n_patterns)` | Binary pattern-presence matrix without running prediction. Use to plug the pattern matrix into a custom downstream model. |
| `get_pattern_info()` | `pd.DataFrame` | One row per mined pattern with columns `pattern`, `utility`, `information_gain`, and `support`. |

#### Notes

- `HUGIMLClassifierPy` does not create or use the `outputs/` directory.
- Parameters `dsName`, `foldNo`, `imbWeights`, `huiItemsPercent`, and `fsK` are available only in the Java-backed `HUGIMLClassifier` and have no equivalent here.
- For datasets with more than ~50000 training rows and `L ≥ 2`, the Java-backed `HUGIMLClassifier` is recommended for performance.

<br/>

### 5. C++ ACCELERATED BACKEND (HUGIMLClassifierNative)

`HUGIMLClassifierNative` is a high-performance implementation of the HUG-IML classifier backed by a compiled C++ extension (`_hugiml_core`) built with [pybind11](https://pybind11.readthedocs.io). It delivers the same sklearn-compatible public API and produces numerically identical results while achieving substantially faster fit and prediction times, especially on large datasets.

The source files are located in the `HUGIMLClassifierCpp/` folder of this repository.

#### 5.1 What runs in C++

All computationally intensive stages execute inside the compiled extension:

- **Discretisation** — quantile binning (`_kbins`), supervised bin-count selection (`_choose_nb`), normalised entropy, information gain, NMI, and Pearson correlation
- **Transaction construction** — three-pass pipeline that discretises columns, assigns utility weights, and builds the utility-annotated transaction list
- **Top-K HUI mining** — utility-list construction, EUCS and LIU pruning, information-gain filtering, bounded min-heap management
- **Binary feature-matrix assembly** — training matrix and test-time transform

Column-type detection, NaN imputation, the downstream sklearn estimator, and all explanation methods remain in Python.

#### 5.2 Prerequisites

A C++17-capable compiler is required:

| Platform | Recommended toolchain |
|---|---|
| Linux | GCC ≥ 7 (`apt install g++`) |
| macOS | Xcode Command Line Tools (`xcode-select --install`) |
| Windows | Visual Studio Build Tools 2017 or later |

Python ≥ 3.9 and the following packages are also needed:

```
numpy>=1.22  pandas>=1.4  scipy>=1.8  scikit-learn>=1.0  pybind11>=2.10
```

#### 5.3 Installation

From inside the `HUGIMLClassifierCpp/` folder, run:

```
pip install -e .
```

This compiles `_hugiml_core` (produces `_hugiml_core.so` on Linux/macOS or `_hugiml_core.pyd` on Windows) and makes both `_hugiml_core` and `HUGIMLClassifierNative` importable from anywhere in your Python environment.

#### 5.4 Command-line usage

Run the sample script from inside the `HUGIMLClassifierCpp/` folder:

```
python HUGIMLClassifierNativeSample.py
```

The sample script runs two benchmark datasets (Pima Indians Diabetes and Titanic) under multiple parameter configurations and prints classification metrics, pattern tables, and cross-validation scores.

#### 5.5 Parameters

The parameter set is identical to `HUGIMLClassifierPy`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `B` | int | `5` | Quantile bins per numerical column. Use `-1` for supervised auto-selection (maximises per-column information gain over [2, 20]). |
| `L` | int | `1` | Maximum pattern length. `1` = singletons; `2` = singletons and pairs; `-1` = unlimited. |
| `G` | float | `1e-4` | Minimum information-gain threshold. Patterns below this are discarded. |
| `topK` | int | `-1` | Maximum patterns to retain. `-1` auto-computes as C(100, L). |
| `allCols` | list | `None` | `[int_cols, float_cols, cat_cols]` — column names grouped by type. Pair with `origColumns`. |
| `origColumns` | list | `None` | Ordered list of all column names matching the columns of X. Pair with `allCols`. |
| `base_estimator` | estimator | `None` | Downstream sklearn classifier. Defaults to `LogisticRegression`. |
| `verbose` | bool | `False` | Print progress during fit. |

#### 5.6 Quick start

**Path A — `prepareXy`** (recommended when the full dataset is available upfront):

```python
from HUGIMLClassifierNative import HUGIMLClassifierNative
from sklearn.model_selection import train_test_split

clf = HUGIMLClassifierNative(B=7, L=1, G=5e-3)
X, y = clf.prepareXy(X_df, y_series)          # auto-detects column types
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y)
clf.fit(X_tr, y_tr)

y_pred_proba = clf.predict_proba(X_te)
y_pred       = y_pred_proba.argmax(axis=1)

print(clf.get_hug_features())   # e.g. ['glucose=[107,140]', 'bmi=[0.275,0.383]']
print(clf.get_pattern_info())   # utility / information gain / support per pattern
```

**Path B — `allCols + origColumns`** (use inside cross-validation loops):

```python
import numpy as np
from HUGIMLClassifierNative import HUGIMLClassifierNative
from sklearn.model_selection import StratifiedKFold, cross_validate

numericIntCols   = [c for c in X.columns if np.issubdtype(X[c].dtype, np.integer)]
numericFloatCols = [c for c in X.columns if np.issubdtype(X[c].dtype, float)]
catCols          = [c for c in X.columns if X[c].dtype == object]

clf = HUGIMLClassifierNative(
    B=-1, L=1, G=1e-6,
    allCols=[numericIntCols, numericFloatCols, catCols],
    origColumns=X.columns.tolist(),
)

cv = cross_validate(
    clf, X, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    scoring=['accuracy', 'f1', 'roc_auc'],
)
```

#### 5.7 Output methods

| Method | Returns | Description |
|---|---|---|
| `get_hug_features()` | `list[str]` | Human-readable label per mined pattern, e.g. `'glucose=[107,140]'` or `'sex=male'`. |
| `get_pattern_info()` | `pd.DataFrame` | One row per pattern with columns `pattern`, `utility`, `information_gain`, `support`. |
| `transform(X)` | `csr_matrix (n, n_patterns)` | Binary pattern-presence matrix without running prediction. Plug into a custom downstream model. |

#### 5.8 Repository files

```
HUGIMLClassifierCpp/
├── setup.py                        Build configuration
├── pyproject.toml                  PEP 517/518 build metadata
├── requirements.txt                Python dependencies
├── HUGIMLClassifierNative.py       sklearn-compatible Python wrapper
├── HUGIMLClassifierNativeSample.py Sample script (Pima + Titanic benchmarks)
├── datasets/
│   ├── pima indians diabetes.csv   Pima Indians Diabetes dataset
│   └── titanic.csv                 Titanic dataset
└── src/
    └── hugiml_core.cpp             C++ extension source (~1 100 lines)
```

#### 5.9 Notes

- `HUGIMLClassifierNative` does not create or use the `outputs/` directory.
- Parameters `dsName`, `foldNo`, `imbWeights`, `huiItemsPercent`, and `fsK` are specific to the Java-backed `HUGIMLClassifier` and have no equivalent here.
- The C++ implementation reproduces the Python version's results to floating-point precision. Prediction probabilities are numerically identical across all tested datasets and parameter combinations.

<br/>