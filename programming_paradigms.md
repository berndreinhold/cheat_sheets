# Programming paradigms
Overview: https://towardsdatascience.com/what-is-a-programming-paradigm-1259362673c2

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

- generics (introduced by C++): templates

## functional programming
- according to: http://blog.cleancoder.com/uncle-bob/2018/04/13/FPvsOO.html
- ```f(a) == f(b) if a == b``` (referential transparency)
- not to mention outside world (persisting in databases, files, etc)
- no assignment operator
- key: recursion
- a function that takes a state data structure as argument. It calls itself and modifies the state data structure, returning it at the end.
- "referential transparency - no reassignment of values"
- FP and OO work nicely together

### stateless
- excellent for multithreading and concurrent code
- no race conditions
- simplified debugging
- https://stackoverflow.com/questions/844536/advantages-of-stateless-programming#844548

### [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)
- iterators as an important building block


## SOLID design principles
for object-oriented programming. 
- **S**ingle Responsibility Principle: group things that change for the same reasons, split things that change for different reasons. Only do one thing.
Ideally results in [less merge conflicts](https://www.freecodecamp.org/news/solid-principles-explained-in-plain-english/).
- **O**pen-Closed Principle: open for enhancement, closed for modification (backwards compatibility)
- **L**iskov Substitution Principle: if S is a subtype of T, feeding S into a program expecting T, the program should behave gracefully. (After Barbara Liskov)
- **I**nterface Segregation Principle: separating interfaces: "Keep interfaces small" so that users do not dependent on things they don't need. (http://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html)
- **D**ependency Inversion Principle: classes should depend on abstract classes and interfaces, rather than concrete classes. "Depend in the direction of abstraction. High-level classes should not depend on low-level details." ([good example](https://hackernoon.com/solid-principles-simple-and-easy-explanation-f57d86c47a7f))


"software hasn’t change all that much since 1945 when Turing wrote the first lines of code for an electronic computer" (http://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html)

### disagreement by Dan North: "write simple code"
- "is-a" and "has-a" mindset (from https://speakerdeck.com/tastapod/why-every-element-of-solid-is-wrong?slide=11)
- "acts-like-a" and "can-be-used-as-a" - composition is easier than inheritance, try to avoid object hierarchies
- "write code, that fits in your head"
- Uncle Bob: he agrees on simple code, his SOLID principles result in simple code.


## Zen of Python
```
$ import this

The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

The relevant ones for me:
- "simple is better than complex"
- "altough practicality beats purity"
- "now is better than never."
- "In the face of ambiguity, refuse the temptation to guess"

## design patterns
- https://www.geeksforgeeks.org/design-patterns-set-1-introduction/
- creational
- structural
- behavioral

## Test Driven Development
