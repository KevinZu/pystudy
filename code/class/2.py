#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 父类
class AddBook(object):
    def __init__(self, name, phone):
        self.name = name
        self.phone = phone

    def get_phone(self):
        return self.phone

# 子类，继承
class EmplEmail(AddBook):
    def __init__(self, nm, ph, email):
        # AddBook.__init__(self, nm, ph) # 调用父类方法一
        super(EmplEmail, self).__init__(nm, ph) # 调用父类方法二
        self.email = email

    def get_email(self):
        return self.email

# 调用
if __name__ == "__main__":
    Detian = AddBook('handetian', '18210413001')
    Meng = AddBook('shaomeng', '18210413002')

    print( Detian.get_phone())
    print( AddBook.get_phone(Meng))

    alice = EmplEmail('alice', '18210418888', 'alice@xkops.com')
    print( alice.get_email(), alice.get_phone())
