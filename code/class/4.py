#!/usr/bin/env python
#coding:utf8

class Books(object):
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        return self.title

    def __repr__(self):
        return self.title

    def __call__(self):
        print( "%s is written by %s" %(self.title, self.author))


if __name__ == '__main__':
    pybook = Books('Core Python', 'Wesley')
    print (pybook)
    pybook()
