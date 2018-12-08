#!/usr/bin/env python
#coding:utf8

class  Number(object):
    """Custum object
    add/radd -> +; 
    sub/rsub -> -;
    mul/rmul -> *;
    div/rdiv -> /;
    """
    def __init__(self, number):
        self.number = number

    def __add__(self, other):
        return self.number + other        

    def __radd__(self, other):
        return self.number  + other

    def __sub__(self, other):
        return self.number - other

    def __rsub__(self, other):
        return other - self.number


    def __gt__(self, other):
        if self.number > other:
            return True
        return False


if __name__ == '__main__':
    num = Number(10)
    print( num + 20)
    print( 30 + num)
    print( num - 5)
    print( 11 - num)
    print( num > 20)
