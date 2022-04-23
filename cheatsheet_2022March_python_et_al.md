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

## matplotlib - events 
onpick()

https://matplotlib.org/stable/users/explain/event_handling.html

## class
class X():
    def Y(self, dataset : str):        
        pass

    def Y(self):
        pass

this does not work, the second Y() masks the first Y() - in C++ one could overload like this, but not in python it seems.

one could use decorators though: 

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

duplicates (...)


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

# VS code
https://stackoverflow.com/questions/51244223/visual-studio-code-how-debug-python-script-with-arguments#51244649
launch.json and args-dictionary