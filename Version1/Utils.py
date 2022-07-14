# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:29:03 2019

@author: Jimena Ferreira - FIng- UdelaR
"""
__all__ = ['tree2symb', 'round_expr', 'Prediction']

def tree2symb(ind):
    from sympy import symbols, sympify
    from sympy import Add, Mul, log, cos, sin, exp, Abs, sqrt
    from sympy import Lambda
    
    X0 = symbols("X0", real=True)
    X1 = symbols("X1", real=True)    
    X2 = symbols("X2", real=True) 
    X3 = symbols("X3", real=True) 
    X4 = symbols("X4", real=True) 
    X5 = symbols("X5", real=True) 
    X6 = symbols("X6", real=True) 
    X7 = symbols("X7", real=True) 
    X8 = symbols("X8", real=True) 
    X9 = symbols("X9", real=True) 
    X10 = symbols("X10", real=True) 
    X11 = symbols("X11", real=True)  
    X12 = symbols("X12", real=True) 
    X13 = symbols("X13", real=True) 
    X14 = symbols("X14", real=True) 
    X15 = symbols("X15", real=True) 
    X16 = symbols("X16", real=True) 
    X17 = symbols("X17", real=True) 
    X18 = symbols("X18", real=True) 
    x=symbols('x', real=True)
    y=symbols('y', real=True)

    locals = {
        "add": Add,
        "mul": Mul,
        "div": Lambda((x, y), x/y),
        "protectedDiv": Lambda((x, y), x/y),
        "sub": Lambda((x,y), x-y),
        "neg": Lambda((x),(-1*x)),
        "log": Lambda((x),log(x)),
        "protectedLog": Lambda((x),log(x)),
        "cos": Lambda((x),cos(x)),
        "sin": Lambda((x),sin(x)),
        "exp": Lambda((x),exp(x)),
        "protectedExp": Lambda((x),exp(x)),
        "abs": Lambda((x),Abs(x)),
        "protectedSqrt":  sqrt,
        "sqrt":  sqrt,
        "square":Lambda((x), x**2),
        "powerReal": Lambda((x, y), x ** y),
        "X0": X0, 
        "X1": X1, 
        "X2": X2, 
        "X3": X3, 
        "X4": X4, 
        "X5": X5, 
        "X6": X6, 
        "X7": X7, 
        "X8": X8, 
        "X9": X9, 
        "X10": X10, 
        "X11": X11, 
        "X12": X12, 
        "X13": X13, 
        "X14": X14, 
        "X15": X15, 
        "X16": X16, 
        "X17": X17, 
        "X18": X18
    }
    final=sympify(ind, locals=locals)
    return final

def round_expr(expr, num_digits):
    from sympy import Float, Number
    expr=expr.xreplace({n : Float(n,num_digits ) for n in expr.atoms(Number)}) 

    return expr

def Prediction(expr, x):
    from sympy import symbols, lambdify
    import numpy as np
    if x.shape[1]==1:
        X0 = symbols("X0", real=True)        
        f = lambdify(X0, expr, "numpy")
        y_est= [f(x[pos,0] ) for pos in range(x.shape[0])]
    elif x.shape[1]==2:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)
        f = lambdify([X0,X1], expr, "numpy")  # "math", "mpmath", "sympy", "numpy"
        y_est= [f(x[pos,0], x[pos,1] ) for pos in range(x.shape[0])]
    elif x.shape[1]==3:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)         
        X2 = symbols("X2", real=True)
        f = lambdify([X0,X1,X2], expr, "numpy")
        y_est= [f(x[pos,0], x[pos,1],  x[pos,2] ) for pos in range(x.shape[0])]
    elif x.shape[1]==4:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)         
        X2 = symbols("X2", real=True)        
        X3 = symbols("X3", real=True)
        f = lambdify([X0,X1,X2,X3], expr, "numpy")
        y_est= [f(x[pos,0], x[pos,1],  x[pos,2], x[pos,3] ) for pos in range(x.shape[0])]
    elif x.shape[1]==5:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)         
        X2 = symbols("X2", real=True)        
        X3 = symbols("X3", real=True)       
        X4 = symbols("X4", real=True)
        f = lambdify([X0,X1,X2,X3,X4], expr, "numpy")
        y_est= [f(x[pos,0], x[pos,1],  x[pos,2], x[pos,3],x[pos,4] ) for pos in range(x.shape[0])]
    elif x.shape[1]==6:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)         
        X2 = symbols("X2", real=True)        
        X3 = symbols("X3", real=True)       
        X4 = symbols("X4", real=True)      
        X5 = symbols("X5", real=True)
        f = lambdify([X0,X1,X2,X3,X4,X5], expr, "numpy")
        y_est= [f(x[pos,0], x[pos,1],  x[pos,2], x[pos,3],x[pos,4],x[pos,5] ) for pos in range(x.shape[0])]
    elif x.shape[1]==7:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)         
        X2 = symbols("X2", real=True)        
        X3 = symbols("X3", real=True)       
        X4 = symbols("X4", real=True)      
        X5 = symbols("X5", real=True)    
        X6 = symbols("X6", real=True)
        f = lambdify([X0,X1,X2,X3,X4,X5,X6], expr, "numpy")
        y_est= [f(x[pos,0], x[pos,1],  x[pos,2], x[pos,3],x[pos,4],x[pos,5],x[pos,6] ) for pos in range(x.shape[0])]
    elif x.shape[1]==8:
        X0 = symbols("X0", real=True)  
        X1 = symbols("X1", real=True)         
        X2 = symbols("X2", real=True)        
        X3 = symbols("X3", real=True)       
        X4 = symbols("X4", real=True)      
        X5 = symbols("X5", real=True)    
        X6 = symbols("X6", real=True)   
        X7 = symbols("X7", real=True)
        f = lambdify([X0,X1,X2,X3,X4,X5,X6,X7], expr, "numpy") 
        y_est= [f(x[pos,0], x[pos,1], x[pos,2], x[pos,3],x[pos,4],x[pos,5],x[pos,6],x[pos,7] ) for pos in range(x.shape[0])]
    elif x.shape[1] == 9:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7,X8], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]) for pos in
                 range(x.shape[0])]
    elif x.shape[1] == 10:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9]) for pos in range(x.shape[0])]
    elif x.shape[1] == 11:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10], expr, "numpy") 
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10]) for pos in range(x.shape[0])]
    elif x.shape[1] == 12:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11], expr, "numpy") 
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11]) for pos in range(x.shape[0])]
    elif x.shape[1] == 13:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        X12 = symbols("X12", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12], expr, "numpy") 
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11], x[pos, 12]) for pos in range(x.shape[0])]
    elif x.shape[1] == 14:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        X12 = symbols("X12", real=True)
        X13 = symbols("X13", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11], x[pos, 12], x[pos, 13]) for pos in range(x.shape[0])]
    elif x.shape[1] == 15:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        X12 = symbols("X12", real=True)
        X13 = symbols("X13", real=True)
        X14 = symbols("X14", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11], x[pos, 12], x[pos, 13], x[pos, 14]) for pos in range(x.shape[0])]
    elif x.shape[1] == 16:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        X12 = symbols("X12", real=True)
        X13 = symbols("X13", real=True)
        X14 = symbols("X14", real=True)
        X15 = symbols("X15", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11], x[pos, 12], x[pos, 13], x[pos, 14], x[pos, 15]) for pos in range(x.shape[0])]
    elif x.shape[1] == 17:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        X12 = symbols("X12", real=True)
        X13 = symbols("X13", real=True)
        X14 = symbols("X14", real=True)
        X15 = symbols("X15", real=True)
        X16 = symbols("X16", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11], x[pos, 12], x[pos, 13], x[pos, 14], x[pos, 15], x[pos, 16])
                 for pos in range(x.shape[0])]
    elif x.shape[1] == 18:
        X0 = symbols("X0", real=True)
        X1 = symbols("X1", real=True)
        X2 = symbols("X2", real=True)
        X3 = symbols("X3", real=True)
        X4 = symbols("X4", real=True)
        X5 = symbols("X5", real=True)
        X6 = symbols("X6", real=True)
        X7 = symbols("X7", real=True)
        X8 = symbols("X8", real=True)
        X9 = symbols("X9", real=True)
        X10 = symbols("X10", real=True)
        X11 = symbols("X11", real=True)
        X12 = symbols("X12", real=True)
        X13 = symbols("X13", real=True)
        X14 = symbols("X14", real=True)
        X15 = symbols("X15", real=True)
        X16 = symbols("X16", real=True)
        X17 = symbols("X17", real=True)
        f = lambdify([X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17], expr, "numpy")  
        y_est = [f(x[pos, 0], x[pos, 1], x[pos, 2], x[pos, 3], x[pos, 4], x[pos, 5], x[pos, 6], x[pos, 7], x[pos, 8]
                   , x[pos, 9], x[pos, 10], x[pos, 11], x[pos, 12], x[pos, 13], x[pos, 14], x[pos, 15], x[pos, 16], x[pos, 17])
                 for pos in range(x.shape[0])]

    y_est=np.asarray(y_est).reshape((-1,1))
    y_est=np.nan_to_num(y_est)
    return y_est