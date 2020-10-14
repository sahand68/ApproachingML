def print_directory_contents(sPath):
    """
    This function takes the name of a directory 
    and prints out the paths files within that 
    directory as well as any files contained in 
    contained directories. 

    This function is similar to os.walk. Please don't
    use os.walk in your answer. We are interested in your 
    ability to work with nested structures. 
    """
    import os

    for sChild in os.listdir(sPath):
        sChildPath= os.path.join(sPath, sChild)
        if os.path.isdir(sChildPath) :
            print_directory_contents(sChildPath)
        else:
            print(sChildPath)




A0 = dict(zip(('a','b','c','d','e'),(1,2,3,4,5)))
A1 = range(10)
A2 = sorted([i for i in A1 if i in A0])
A3 = sorted([A0[s] for s in A0])
A4 = [i for i in A1 if i in A3]
A5 = {i:i*i for i in A1}
A6 = [[i,i*i] for i in A1]


def f(*args,**kwargs): print(args, kwargs)

l = [1,2,3]
t = (4,5,6)
d = {'a':7,'b':8,'c':9}

# f()
# f(1,2,3)                    # (1, 2, 3) {}
# f(1,2,3,"groovy")           # (1, 2, 3, 'groovy') {}
# f(a=1,b=2,c=3)              # () {'a': 1, 'c': 3, 'b': 2}
# f(a=1,b=2,c=3,zzz="hi")     # () {'a': 1, 'c': 3, 'b': 2, 'zzz': 'hi'}
# f(1,2,3,a=1,b=2,c=3)        # (1, 2, 3) {'a': 1, 'c': 3, 'b': 2}

# f(*l,**d)                   # (1, 2, 3) {'a': 7, 'c': 9, 'b': 8}
# f(*t,**d)                   # (4, 5, 6) {'a': 7, 'c': 9, 'b': 8}
# f(1,2,*t)                   # (1, 2, 4, 5, 6) {}
# f(q="winning",**d)          # () {'a': 7, 'q': 'winning', 'c': 9, 'b': 8}
# f(1,2,*t,q="winning",**d)   # (1, 2, 4, 5, 6) {'a': 7, 'q': 'winning', 'c': 9, '


def f2(arg1,arg2,*args,**kwargs): print(arg1,arg2, args, kwargs)



f2(1,2,3)                       # 1 2 (3,) {}
# f2(1,2,3,"groovy")              # 1 2 (3, 'groovy') {}
# f2(arg1=1,arg2=2,c=3)           # 1 2 () {'c': 3}
# f2(arg1=1,arg2=2,c=3,zzz="hi")  # 1 2 () {'c': 3, 'zzz': 'hi'}
# f2(1,2,3,a=1,b=2,c=3)           # 1 2 (3,) {'a': 1, 'c': 3, 'b': 2}

# f2(*l,**d)                   # 1 2 (3,) {'a': 7, 'c': 9, 'b': 8}
# f2(*t,**d)                   # 4 5 (6,) {'a': 7, 'c': 9, 'b': 8}
# f2(1,2,*t)                   # 1 2 (4, 5, 6) {}
# f2(1,1,q="winning",**d)      # 1 1 () {'a': 7, 'q': 'winning', 'c': 9, 'b': 8}
# f2(1,2,*t,q="winning",**d)










if __name__ == "__main__":

#     print_directory_contents(sPath)
    # print(f'A0 = {A0}')
    # print(f'A1 = {A1}')
    # print(f'A2 = {A2}')
    # print(f'A3 = {A3}')
    # print(f'A4 = {A4}')
    # print(f'A5 = {A5}')
    # print(f'A6 = {A6}')

    print(f2(1,2,3))                     # 1 2 (3,) {}
    # print(f2(1,2,3,"groovy"))             # 1 2 (3, 'groovy') {}
    # print(f2(arg1=1,arg2=2,c=3))           # 1 2 () {'c': 3}
    # print(f2(arg1=1,arg2=2,c=3,zzz="hi"))  # 1 2 () {'c': 3, 'zzz': 'hi'}
    # print(f2(1,2,3,a=1,b=2,c=3))           # 1 2 (3,) {'a': 1, 'c': 3, 'b': 2}

    # print(f2(*l,**d))                   # 1 2 (3,) {'a': 7, 'c': 9, 'b': 8}
    # print(f2(*t,**d))                  # 4 5 (6,) {'a': 7, 'c': 9, 'b': 8}
    # print(f2(1,2,*t))                  # 1 2 (4, 5, 6) {}
    # print(f2(1,1,q="winning",**d))      # 1 1 () {'a': 7, 'q': 'winning', 'c': 9, 'b': 8}
    # print(f2(1,2,*t,q="winning",**d))