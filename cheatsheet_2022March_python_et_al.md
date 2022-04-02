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

## class
class X():
    def Y(self, dataset : str):        
        pass

    def Y(self):
        pass

this does not work, the second Y() masks the first Y()

# markdown


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
