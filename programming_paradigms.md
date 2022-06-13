# Programming paradigms


## polymorphism 
- german: Vielgestaltigkeit (auch bekannt in der Biologie)
- function or operator overloading
- same function, different parameter types and return values. Different procedures/ operations depending on the argument types. From a compiler perspective very different functions.
- http://blog.cleancoder.com/uncle-bob/2018/04/13/FPvsOO.html
o.f() vs f(o) - no semantic difference, syntax difference 
The former means there is an object o, which calls a function f() and what f() does depends on the object.

- OO without polymorphism is no OO.
- reductive definition of OO by uncle bob:
    - the paradigm to have polymorphism in function calls, that do not include a dependency of the caller (calling function) to the callee (the argument of the calling function).


## object oriented programming
- polymorphism
- inheritance
- encapsulation

- private, protected, public methods (C++)
- python: everything is public
- "“Private” instance variables that cannot be accessed except from inside an object don’t exist in Python. However, there is a convention that is followed by most Python code: a name prefixed with an underscore (e.g. _spam) should be treated as a non-public part of the API (whether it is a function, a method or a data member). It should be considered an implementation detail and subject to change without notice." (from https://docs.python.org/3/tutorial/classes.html#tut-private)

## functional programming
- according to: 


## SOLID design principles