# python
## pandas
    print(type(df_duplicates_only[["diff_sgv_mean", "diff_sgv_std"]]))  # a dataframe
    print(type(df_duplicates_only["diff_sgv_mean"]))  # a series


When mask is a boolean NumPy array, df.loc[mask] selected rows where the mask is True. If mask is a DataFrame, however, then df.loc[mask] selects rows from df whose index value matches the index value in mask which corresponds to a True value. This alignment of indices is wonderful when you need it, but slows down performance when you don't.
https://stackoverflow.com/questions/26640129/search-for-string-in-all-pandas-dataframe-columns-and-filter#26641085

_code does not do what I like it to do :)_
_column of group by statement is odd_


    self.df[f"{ds}_groupby"] = self.df[ds].groupby(["date", "user_id"], as_index=False, dropna=False).apply(lambda x: f", {len(x.filename)}" + ", ".join(x.filename))
    # .agg("count")
    print(f"{ds} after groupby (date, user_id): ", len(self.df[f"{ds}_groupby"]))
    self.df[f"{ds}_groupby"].columns = ["date", "user_id", "filenames"]
    # print(self.df[f"{ds}_groupby"])
    # gui = pdg.show(self.df[f"{ds}_groupby"])

    test = self.df[f"{ds}_groupby"]
    print(test)
    test = test.loc[test["filenames"].apply(lambda a: '"' in str(a))]
    print(test["filenames"].apply(lambda a: '"' in str(a)))

## pandas pipe
- _looks very cool_
- https://towardsdatascience.com/the-unreasonable-effectiveness-of-method-chaining-in-pandas-15c2109e3c69
- https://tomaugspurger.github.io/method-chaining  has also some very nice visualisation calls

## pandas dataframe
- has no attribute unique(): df.unique(), but pd.Series has a unique()-function
- dataframe.drop_duplicates()

## pandas groupby, sort, agg
    self.df[f"{dataset}_groupby"] = self.df[dataset][["date", "user_id", "filename"]].groupby(["date", "user_id"], as_index=False, dropna=False).agg("count")
    print(self.df[f"{dataset}_groupby"][self.df[f"{dataset}_groupby"]["filename"]> 1])
    self.df[f"{dataset}_groupby"] = self.df[dataset][["date", "user_id", "filename"]].groupby(["date", "user_id"], as_index=False, dropna=False).agg(count_filename= ('filename', 'count'), groupby_filename=('filename', lambda x: ", ".join(x)))
    # self.df[f"{dataset}_groupby"] = self.df[dataset].groupby(["date", "user_id"], as_index=False, dropna=False).apply(lambda x: f"{len(x.filename)}, " + ", ".join(x.filename))
    # .agg("count")
    print(f"{dataset} after groupby (date, user_id): ", len(self.df[f"{dataset}_groupby"]))
    #self.df[f"{dataset}_groupby"].columns = ["date", "user_id", "sgv_mean","sgv_std", "sgv_min", "sgv_max", "sgv_count", "filenames"]
    #self.df[f"{dataset}_groupby"].to_csv(f"{dataset}_groupby_date_user_id.csv")
    print(self.df[f"{dataset}_groupby"])
    # gui = pdg.show(self.df[f"{dataset}_groupby"])
    df = self.df[f"{dataset}_groupby"][self.df[f"{dataset}_groupby"]["count_filename"]> 1]
    df.sort_values(by=['user_id', 'date']).to_csv("test.csv")

    df_gb = self.df[dataset][["date", "user_id", "filename"]].groupby(["date", "user_id"], as_index=False, dropna=False).agg(count_filename= ('filename', 'count'), groupby_filename=('filename', lambda x: x)) # either a StringArray or a string

    df_gb = df_gb.apply({'groupby_filename': lambda x: x.split(",")[0]})  # before df_gb has 4 columns, after this apply() only the groupby_filename column - why?

    df["weather"].value_counts()  # somewhat similar to groupby-count statements.


## pandas dates:
    if self.df[ds]["date"].min() is not pd.NaT:
        self.min_date = min([self.df[ds]["date"].min(), self.min_date])
    if self.df[ds]["date"].max() is not pd.NaT:
        self.max_date = max([self.df[ds]["date"].max(), self.max_date])

need to check for pd.NaT, because it would both be min and max.


tried many different things:
        pd.Timedelta() works best for adding margins to dates for plotting. 
        add 5 % margin to the 
        plt.xlim((self.min_date - pd.Timedelta(10, "days"), self.max_date + pd.Timedelta(10, "days")))
        
        print("plot: ", self.min_date, self.max_date)
        dates_with_margin = (self.min_date.value*0.99, self.max_date.value*1.01)  # 1 percent margin, .value gives the unix timestamp in nanosec
        print("plot: ", dates_with_margin)
        dates_with_margin2 = (pd.to_datetime(dates_with_margin[0], unit="ns"), pd.to_datetime(dates_with_margin[1], unit="ns"))
        dates_with_margin3 = 
        print("plot: dates_with_margin2: ", dates_with_margin2)
        print("plot: dates_with_margin3: ", dates_with_margin3)
        
## pandas fill variable depending on condition, that features other columns:
        out.loc[pd.isnull(out["user_id_ds2"]), "dataset"] = 1  # ds1
        out.loc[pd.isnull(out["user_id_ds1"]), "dataset"] = 3  # ds2
        out.loc[~(pd.isnull(out["user_id_ds2"]) | pd.isnull(out["user_id_ds1"])), "dataset"] = 2  # duplicates
        out["dataset"] = out["dataset"].astype(int)

## pandas .loc and row data structure
one row as selected through: df.loc[[selection-criteria]].apply(lambda row: print(row))
user_id_ds1_x     NaN
user_id_ds2       NaN
user_id_ds3_x    38.0
user_id_ds1_y     NaN
user_id_ds3_y     NaN
Name: 10, dtype: object

Name is the index


## pandas .loc, apply and lambda
```python
def get_the_right_value2(row, col_names : str):
        x = row[col_names[0]]
        y = row[col_names[1]]

        # it is clear already from the dataframe selection that one of x or y are not NA
        if pd.isna(x): return y
        elif pd.isna(y): return x
        else:
            if x == y: return x
            else: raise ValueError(f"{x} or {y} are not identical, even though they should be, if they are both not NA. row: {row}")
            
    #df_merged.loc[(~pd.isna(df_merged["user_id_ds3_x"]) |  ~pd.isna(df_merged["user_id_ds3_y"])), "user_id_ds3"] = df_merged.loc[(~pd.isna(df_merged["user_id_ds3_x"]) |  ~pd.isna(df_merged["user_id_ds3_y"])), ["user_id_ds3_x", "user_id_ds3_y"]].apply(lambda x: get_the_right_value(x[0],x[1]), axis=1)
    df_merged.loc[(~pd.isna(df_merged["user_id_ds3_x"]) |  ~pd.isna(df_merged["user_id_ds3_y"])), "user_id_ds3"] = df_merged.loc[(~pd.isna(df_merged["user_id_ds3_x"]) |  ~pd.isna(df_merged["user_id_ds3_y"]))].apply(lambda row: get_the_right_value2(row, ["user_id_ds3_x", "user_id_ds3_y"]), axis=1)
```
from process_test_datasets.py

https://realpython.com/python-lambda/

## time series
https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py

```python
from sklearn.model_selection import TimeSeriesSplit
```

### interesting Concluding remarks

"We note that we could have obtained slightly better results for kernel models by using more components (higher rank kernel approximation) at the cost of longer fit and prediction durations. For large values of n_components, the performance of the one-hot encoded features would even match the spline features.

The Nystroem + RidgeCV regressor could also have been replaced by MLPRegressor with one or two hidden layers and we would have obtained quite similar results.

The dataset we used in this case study is sampled on a hourly basis. However cyclic spline-based features could model time-within-day or time-within-week very efficiently with finer-grained time resolutions (for instance with measurements taken every minute instead of every hours) without introducing more features. One-hot encoding time representations would not offer this flexibility.

Finally, in this notebook we used RidgeCV because it is very efficient from a computational point of view. However, it models the target variable as a Gaussian random variable with constant variance. For positive regression problems, it is likely that using a Poisson or Gamma distribution would make more sense. This could be achieved by using GridSearchCV(TweedieRegressor(power=2), param_grid({"alpha": alphas})) instead of RidgeCV."
from https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html#sphx-glr-auto-examples-applications-plot-cyclical-feature-engineering-py


# python reduce()
- https://realpython.com/python-reduce-function/
- reduce() performs an operation called folding or reduction.
- aggregates an array to one value


```
def reduce(function, iterable, initializer=None):
    it = iter(iterable)
    if initializer is None:
        value = next(it)
    else:
        value = initializer
    for element in it:
        value = function(value, element)
    return value
```

"In functional programming, fold (also termed reduce, accumulate, aggregate, compress, or inject) refers to a family of higher-order functions that analyze a recursive data structure and through use of a given combining operation, recombine the results of recursively processing its constituent parts, building up a return value. Typically, a fold is presented with a combining function, a top node of a data structure, and possibly some default values to be used under certain conditions. The fold then proceeds to combine elements of the data structure's hierarchy, using the function in a systematic way." (from https://en.wikipedia.org/wiki/Fold_%28higher-order_function%29)


## matplotlib 
### cheatsheets
- https://github.com/matplotlib/cheatsheets and pdfs therein:
- [matplotlib cheatsheet p.1](https://camo.githubusercontent.com/bc3b143766ed68eb6a851900c317c5d9222eb1471888942afb35137aa5141557/68747470733a2f2f6d6174706c6f746c69622e6f72672f63686561747368656574732f63686561747368656574732d312e706e67)
- [matplotlib cheatsheet p.2](https://camo.githubusercontent.com/8566d191963c2ada58246241d19a1252c519edea1ecf4049f5bc939e302e36a8/68747470733a2f2f6d6174706c6f746c69622e6f72672f63686561747368656574732f63686561747368656574732d322e706e67)
- [matplotlib for beginners](https://camo.githubusercontent.com/b1b8838502a81077591ccadbf45dc45f2207637b41245e557198b680b0a2e662/68747470733a2f2f6d6174706c6f746c69622e6f72672f63686561747368656574732f68616e646f75742d626567696e6e65722e706e67)
- [matplotlib for intermediate users](https://camo.githubusercontent.com/fc055a0d3897e7aec7ec66fc1d7f70cfb2873f82eb5be4ea977286a1cf08fa74/68747470733a2f2f6d6174706c6f746c69622e6f72672f63686561747368656574732f68616e646f75742d696e7465726d6564696174652e706e67)
- [tips & tricks](https://camo.githubusercontent.com/62a744e98372f7aaad377cf1f535dcc10117ff196c876102682b03ca4759f420/68747470733a2f2f6d6174706c6f746c69622e6f72672f63686561747368656574732f68616e646f75742d746970732e706e67)

### events 
onpick()

https://matplotlib.org/stable/users/explain/event_handling.html

## Seaborn
```
sns.distplot(df['Profit'])
plt.title("Distribution of Profit")
sns.despine()
```
from https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1


## statistical measures
### skewness
how asymmetric is a distribution?
### curtosis
how tailed or peaked is a distribution?
c > 3 


## class
class X():
    def Y(self, dataset : str):        
        pass

    def Y(self):
        pass

this does not work, the second Y() masks the first Y() - in C++ one could overload like this, but not in python it seems.

one could use decorators though


### alias to class methods
[alias to private member function](img/python_private_member_function.png)
Includes mangling, avoids conflicts, since the ```__update = update``` seems to be called before it is replaced by the update()-function of the sub class.
It thereby avoids conflicts in ```__init__()```
Also the alias is called without arguments. Very clean and simple.

## named tuples
https://stackoverflow.com/questions/2970608/what-are-named-tuples-in-python
tuples are immutable
named tuples are tuples with additional possibility to access variables by name, rather than index

dicts are mutable and slower than tuples
tuples are immutable, lightweight, but lack readability
namedtuples are the perfect compromise for the two approaches
make it more readable

namedtuple is a class factory

namedtuples are polymorphic
treat them like a class

namedtuple._make()
namedtuple._fields
namedtuple._replace()

Since named tuples are immutable, they can be used as dict keys. 

namedtuple._asdict()



os.makedirs(path, exist_ok=True)

## logging
https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
https://stackoverflow.com/questions/50391429/logging-in-classes-python

## built-in functions
- https://docs.python.org/3/library/functions.html#hasattr
Interesting list of built-in functions, with a focus on hasattr()
- hasattr() allows to check, if a certain function is implemented.
- getattr() call a function: `s = getattr(clf, method)(X[-3:])`
- [nice example](https://gist.github.com/jnothman/4807b1b0266613c20ba4d1f88d0f8cf5)

## performance python code classes
turned code into classes and following that the garbage collection seemed not to work as well anymore. Ran into memory issues with the programs.

## if statements cannot be part of a for loop.
Except use generators or two lines :)
https://stackoverflow.com/questions/6981717/pythonic-way-to-combine-for-loop-and-if-statement

## testing, Test Driven Development (TDD)
https://codefellows.github.io/sea-python-401d2/lectures/tdd_with_pytest.html
Look for python files starting with test_, then write functions starting with test_. These are executed by pytest.

"The solution should be pretty clear. If the problem is that development and testing are too far apart, then move them closer together. This is the aim of Test Driven Development. To close both the temporal and conceptual gaps between the worlds of development and testing."

Is also interesting with datasets, that are being processed. Provide test datasets to run the code with.

https://docs.python.org/3/library/unittest.html#assert-methods

https://docs.pytest.org/en/7.1.x/ (pytest is a bit more lightweight than unittest and comes without API)

https://wiki.python.org/moin/PythonTestingToolsTaxonomy pytest or unittest are good choices

## pyplot
The issue: the outer part of the plot was transparent (alpha=0.0). I needed it opaque (alpha=1.0)
```python
plt.rcParams().update([dict]) 
```
did the job:

```python
plt.rcParams.update({"figure.facecolor" : (1.0, 1.0, 1.0, 1.0), "savefig.facecolor": (1.0, 1.0, 1.0, 1.0)})
plt.figure(figsize=(15,15),facecolor='red')
```
The red facecolor and transparent=False would not work, even without the presence of the ``plt.rcParams.update([dict])`` command.

## Venn diagrams and UpsetPlot
- https://towardsdatascience.com/how-to-create-and-customize-venn-diagrams-in-python-263555527305
- https://pypi.org/project/UpSetPlot/
- https://upsetplot.readthedocs.io/en/latest/api.html
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6876017


## small
- int(x), where x is a tuple does not work element-wise

# markdown
## links to headlines in markdown
- https://docs.gitlab.com/ee/user/markdown.html#header-ids-and-links
- inspect html source (right-click in browser)
- https://stackoverflow.com/questions/51221730/markdown-link-to-header

## automatic TOC on GIThub
https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/

## python documentation generator
comparison: https://medium.com/@peterkong/comparison-of-python-documentation-generators-660203ca3804

recommendation: pdoc for smaller projects


## random number generation
```python
from numpy import random
```
many functions to choose from.

https://stackoverflow.com/questions/17821458/random-number-from-histogram
### KDE - kernel density estimation
https://nbviewer.org/url/mglerner.com/HistogramsVsKDE.ipynb

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
https://www.mvstat.net/tduong/research/seminars/seminar-2001-05/

# JSON
keys must be strings, cannot be integers and should be different from each other

# OS
## usb drive
usb drive could not be mounted anymore in Linux.
in windows it still works
my Linux hat lost power all of sudden, even though plugged in.

try: 
dmesg | less

top
k, 
L, &

# awk
```
reinhold@Eddimed-lnx01:~/Daten/OPEN/OpenAPS_Data/raw$ find . -maxdepth 3 -iname "*direct-sharing-*" | awk -F 'direct-sharing-' '{print $NF}' | sort -n | uniq -c
    288 31
      8 133
     14 351
     78 396

reinhold@Eddimed-lnx01:~/Daten/OPEN/OpenAPS_Data/raw$ find . -maxdepth 3 -iname "*direct-sharing-*" | awk -F 'direct-sharing-' '{print "direct-sharing-"$NF}' | sort -n | uniq -c
      8 direct-sharing-133
    288 direct-sharing-31
     14 direct-sharing-351
     78 direct-sharing-396

reinhold@Eddimed-lnx01:~/Software/OPEN/OPEN_diabetes(classes_config_files)$ awk -F '"' '{print $2}' OPENonOH_groupby_date_user_id_selection.csv | awk -F',' '{print $1}' | sort -n | uniq -c
      1 
   2805 2
     49 3
     77 4
     24 5
     27 6
    165 7
    173 8
```
duplicates (...)

```
reinhold@Eddimed-lnx01:~/Software/OPEN/OPEN_diabetes(classes_config_files)$ awk -F '"' '{print $2}' ds1_groupby_date_user_id.csv | awk -F',' '{print $1}' | sort -n | uniq -c
      1 
  26183 1
   3084 2

reinhold@Eddimed-lnx01:~/Software/OPEN/OPEN_diabetes(classes_config_files)$ awk -F '"' '{print $2}' ds2_groupby_date_user_id.csv | awk -F',' '{print $1}' | sort -n | uniq -c
      1 
  18993 1
   2781 2
     49 3
     77 4
     24 5
     27 6
    165 7
    173 8
```

# VS code
_Disclaimer: below sentences have been copied from the respective documentations. Authorship of these sentences is with the authors of these webpages._ 

```
CTRL + SHFT + P  # Control Palette
```
Python: Select Interpreter
## debugging
https://stackoverflow.com/questions/51244223/visual-studio-code-how-debug-python-script-with-arguments#51244649
launch.json and args-dictionary

https://code.visualstudio.com/docs/python/debugging
launch.json
```
F5, CTRL+SHFT+F5
```
remote debugging: https://code.visualstudio.com/docs/python/debugging#_debugging-by-attaching-over-a-network-connection
SSH forwarding

## VS code: debug ipynb notebooks:
```
pip3 install -U ipykernel
```
restart VS code
(there are some minimum requirements, like VS code version > 1.6)

## Options
```
justMyCode = False  
```
## [Environments](https://code.visualstudio.com/docs/python/environments)
### virtual environments:
To prevent such clutter, developers often create a virtual environment for a project. A virtual environment is a folder that contains a copy (or symlink) of a specific interpreter. When you install into a virtual environment, any packages you install are installed only in that subfolder. When you then run a Python program within that environment, you know that it's running against only those specific packages.

    Note: While it's possible to open a virtual environment folder as a workspace, doing so is not recommended and might cause issues with using the Python extension.

You can configure VS Code to use any Python environment you have installed, including virtual and conda environments. You can also use a separate environment for debugging. For full details, see Environments.

## Environmental variables
https://code.visualstudio.com/docs/python/environments#_environment-variables

- dev.env file
- prod.env file
- key-value pairs of environmental variables

The PYTHONPATH environment variable specifies additional locations where the Python interpreter should look for modules. In VS Code, PYTHONPATH can be set through the terminal settings (terminal.integrated.env.*) and/or within an .env file.


# [Unit Tests](https://code.visualstudio.com/docs/python/testing)
A unit is a specific piece of code to be tested, such as a function or a class. Unit tests are then other pieces of code that specifically exercise the code unit with a full range of different inputs, including boundary and edge cases.

Because unit tests are small, isolated pieces of code (in unit testing you avoid external dependencies and use mock data or otherwise simulated inputs), they're quick and inexpensive to run. This characteristic means that you can run unit tests early and often.


Python tests are Python classes that reside in separate files from the code being tested. Each test framework specifies the structure and naming of tests and test files. Once you write tests and enable a test framework, VS Code locates those tests and provides you with various commands to run and debug them.


## ToDo
1. download framework (e.g. Pytest or Unittest frameworks)
2. create test_xxx.py script ([Create Tests](https://code.visualstudio.com/docs/python/testing#_create-tests))
3. [run tests](https://code.visualstudio.com/docs/python/testing#_run-tests) or even [in parallel](https://code.visualstudio.com/docs/python/testing#_run-tests-in-parallel)
4. understand batch normalization: https://towardsdatascience.com/speeding-up-training-of-neural-networks-with-batch-normalization-29e833260c86?source=read_next_recirc---------0---------------------1796877a_8b80_4082_b6d7_655e033b0eb7-------

# Notebooks
"The support for mixing executable code, equations, visualizations, and rich Markdown makes notebooks useful for breaking down new concepts in a story telling form. This makes notebooks an exceptional tool for educators and students!" [quote](https://code.visualstudio.com/docs/datascience/overview)

Looks interesting: Python Interactive Window: https://code.visualstudio.com/docs/python/jupyter-support-py

# Replace For-Loops with Matrix multiplications
twitter comment by Greg Brockman (added to my bookmark, 15.5.2022)
searched and found: https://medium.com/@aishahsofea/for-loops-vs-matrix-multiplication-ee67868f937
make things much faster.

# Ubuntu
## Ubuntu for IoT
for Cloud and Services

# Development Guidelines
- Flake8: [PEP-8](https://peps.python.org/pep-0008/)
- [The Zen of Python](https://peps.python.org/pep-0020/): ```import this```
- [scikit-learn Developer's Guide](https://scikit-learn.org/stable/developers/)

# Github Copilot
- https://github.com/github/copilot-docs/blob/main/docs/visualstudiocode/gettingstarted.md#getting-started-with-github-copilot-in-visual-studio-code
- https://copilot.github.com/


## Python Data Model
https://docs.python.org/3/reference/datamodel.html

"Objects are Python's abstraction for data. All data is either represented by objects or relations between objects."
Every object has an identity, a type and a value. An object's identity never changes once it has been created. You may think of it as the object's address in memory. The 'is' operator checks the identity of an object; the id() function returns an integer representing the identity of an object."
"For CPython id(x) is the memory address where x is stored."

"Object's type is also unchangeable."

an object's mutability is determined by its type.
An immutable object (tuples) can contain mutable objects (lists).

immutable:
- numbers, strings, tuple 

mutable:
- dict, list

objects are never explicitly destroyed, however when they become unreachable, they might be garbage collected. 

reference counting scheme

files: external resources. Close them with the "with"-statement or "try..finally"-statement

containers: objects with references to other objects

tuples, lists, dictionaries are containers

a=1, b=1 depending on the implementation might point to the same object

### built-in types
None, NotImplemented - single object with that value
Ellipsis: ...

two types of integers: int, boolean

#### sequences
- finite ordered sets, indexed by non-negative numbers
- has len(), items of the sequence are accessable through an index i, s[i]
- supports slicing

##### immutable sequences
- strings, bytes, tuples: the items of tuples are arbitrary Python objects

##### mutable sequences
- lists, dictionaries
- array, collections

##### on dictionaries
- a dictionary is a hash table, not a sequence
- the order of objects in a dicionary is out of the programmer's control.
- the access of objects is by key, never by value: a dictionary is a one-way tool
- keys are case-sensitive
- items(), keys(), values()
- del dict[key]
-

#### sets
- unordered, finite, immutable objects
- iterated over, len()
- not indexable via subscript
- common use cases: remove duplicates from sequences, mathematical operations such as intersect, union, difference, symmetric difference
- for set elements the same immutability rules apply as for dictionary keys (sets are dictionaries with just the keys, without values)

##### sets

##### frozen sets


## Callable types
- ```__func__.__doc__```
- ```__func__.__name__```
- ```__module__```
- ```__globals__```
- ```__defaults__```

## Generator functions
- a function that uses the ```yield``` statement: iterator object

## modules
- import
- a module call is equivalent ```m.x``` to ```m.__dict__["x"]``` 

## 3.3 Special method names
- Python approach to operator overloading
- ```__getitem()__```
- ```x[i]``` is equivalent to ```type(x).__getitem(x, i)__```
- ```__new()__```: customize class creation, in particular for immutable types
- ```__new()__``` is closely related to ```__init()__```
- ```super().__init()__```: call init() of the base class

"Called when the instance is about to be destroyed. This is also called a finalizer or (improperly) a destructor. If a base class has a ```__del__()``` method, the derived class’s ```__del__()``` method, if any, must explicitly call it to ensure proper deletion of the base class part of the instance."
from https://docs.python.org/3/reference/datamodel.html

```del x``` doesn’t directly call ```x.__del__()``` — the former decrements the reference count for x by one, and the latter is only called when x’s reference count reaches zero.

### 3.3.2 Customizing attribute access
None

### Metaclasses

### Ellipsis
https://stackoverflow.com/questions/772124/what-does-the-ellipsis-object-do



# Markdown formatting:
formatting with the depth level:
-1
-2 
or:
- 
-- 
---
- 
--
--

potentially interesting new idea

## Point Cloud Methods
### ICP: Iterative Closest Point
### stochastic ICP: 
### Iterative Random Consensus Projection
https://github.com/ohadmen/pyircp