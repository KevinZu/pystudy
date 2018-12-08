#!/usr/bin/env python
#coding:utf8


class Hotel(object):
    """docstring for Hotel"""
    def __init__(self, room, cf=1.0, br=15):
        self.room = room
        self.cf = cf
        self.br = br

    def cacl_all(self, days=1):
        return (self.room * self.cf + self.br) * days

if __name__ == '__main__':
    stdroom = Hotel(200)
    big_room = Hotel(230, 0.9)
    print( stdroom.cacl_all())
    print( stdroom.cacl_all(2))
    print( big_room.cacl_all())
    print( big_room.cacl_all(3))
