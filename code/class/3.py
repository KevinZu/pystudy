#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
1.class类的组合使用
2.手机、邮箱、QQ等是可以变化的（定义在一起），姓名不可变（单独定义）。
3.在另一个类中引用
'''

class Info(object):
    def __init__(self, phone, email, qq):
        self.phone = phone
        self.email = email
        self.qq = qq

    def get_phone(self):
        return self.phone

    def update_phone(self, newphone):
        self.phone = newphone
        print ("手机号更改已更改")

    def get_email(self):
        return self.email


class AddrBook(object):
    '''docstring for AddBook'''
    def __init__(self, name, phone, email, qq):
        self.name = name
        self.info = Info(phone, email, qq)


if __name__ == "__main__":
    Detian = AddrBook('handetian', '18210413001', 'detian@xkops.com', '123456')
    print( Detian.info.get_phone())
    Detian.info.update_phone(18210413002)
    print( Detian.info.get_phone())
    print( Detian.info.get_email())
