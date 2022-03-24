# python
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

