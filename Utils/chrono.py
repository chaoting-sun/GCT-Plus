# chrono.py
# ---------
# decorator and class to get function/method execution statistics
#
# @Chrono decorator 
# -----------------
# Wraps function/method to track time statistics
#
# @Chrono
# def myFunction(a,b,c):
#     ...
#
# for _ in range(13): myFunction(1,2,3)
#
# myfunction.stats   # prints...
#
# myFunction:
#     Last time:    0.0066 sec.
#     Total time:   0.0941 sec.
#     Call Count:   13
#     Average time: 0.0072 sec.
#     Maximum time: 0.0098 sec.
#     Recursion:    39
#     Max. Depth:   3
#
# myFunction.count     # 13
# myFunction.recursed  # 39 
# myFunction.totalTime # 0.0941
# myFunction.lastTime  # 0.0066
# myFunction.maxDepth  # 3
# myFunction.maxTime   # 0.0098
# myFunction.average   # 0.0072
#
# myFunction.clear()   # resets statistics
# myfunction.depth     # current recursion depth
#
#
# global 'chrono' object 
# ----------------------
# computes time of parameter execution
#
# t,r = chrono.time(len(str(factorial(5000)))) 
#
# returns time and result: t: 0.005331993103027344   r:16326
#
# chrono.print('5000! :',scale=1000)(len(str(factorial(5000))))
# 5.2319 5000! : 16326

from time import time
class Chrono:
     
    def __init__(self,func):
        self.func      = func
        self._name     = None
        self.clear()

    def clear(self):
        self.count     = 0
        self.recursed  = 0
        self.totalTime = 0
        self.start     = None
        self.depth     = -1
        self.lastTime  = None
        self.maxDepth  = 0
        self.maxTime   = 0

    def __call__(self,*args,**kwargs):
        self.depth    += 1
        self.count    += self.depth == 0
        self.recursed += self.depth > 0
        if self.depth == 0:
            self.start     = time()
            
        result = self.func(*args,**kwargs)
        
        if self.depth == 0:
            self.lastTime   = time()-self.start
            self.totalTime += self.lastTime
            self.maxTime    = max(self.lastTime,self.maxTime)
        else:
            self.maxDepth = max(self.maxDepth,self.depth)
        self.depth -= 1
        return result

    @property
    def name(self):
        if self._name is None:
            self._name = ""
            for n,f in globals().items():
                if f is self: self._name = n;break
        return self._name
    
    def methodCaller(self,obj):
        def withObject(*args,**kwargs):       
            return self(obj,*args,**kwargs)  # inject object instance
        return withObject

    def __get__(self,obj,objtype=None):   # return method call or CallCounter
        return self.methodCaller(obj) if obj else self

    @property
    def average(self): return self.totalTime/max(1,self.count)

    @property
    def stats(self):
        print(f"{self.name}:")
        print(f"    Last time:    {self.lastTime:3.4f} sec.")
        print(f"    Total time:   {self.totalTime:5.4f} sec.")
        print(f"    Call Count:   {self.count}")
        print(f"    Average time: {self.average:3.4f} sec.")
        print(f"    Maximum time: {self.maxTime:3.4f} sec.")
        print(f"    Recursion:    {self.recursed}")
        print(f"    Max. Depth:   {self.maxDepth}")

    @property
    def important_stats(self):
        return self.count, self.totalTime

    @property
    def time(self):
        start = time()
        self.count += 1
        def execute(result):
            self.lastTime = time()-start
            self.totalTime += self.lastTime
            return self.lastTime,result
        return execute

    def print(self,label="",scale=1):
        start = time()
        self.count += 1
        def execute(result):
            self.lastTime = time()-start
            self.totalTime += self.lastTime
            print(f"{self.lastTime*scale:3.4f}",label,end=" ")
            if ":" in label:print(result)
            else: print()
            return result
        return execute

chrono = Chrono(None)

if __name__ == '__main__':

    @Chrono
    def myFunction(a,b,c,r=3):
        for _ in range(100000*a): pass
        if r>0: myFunction(1,b,c,r-1)
        return a+b+c

    for i in range(13): myFunction(1,2,3)

    myFunction.stats

"""
myFunction:
    Last time:    0.0062 sec.
    Total time:   0.1005 sec.
    Call Count:   13
    Average time: 0.0077 sec.
    Maximum time: 0.0106 sec.
    Recursion:    39
    Max. Depth:   3
"""