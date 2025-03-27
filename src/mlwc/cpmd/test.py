
def f(x):
    if x<=1:
        return 0
    elif x%2==0:
        return f(x+1)+x
    else:
        return f(x-3)-x



print(f(10))
