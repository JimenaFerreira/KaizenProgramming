# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:29:03 2019

@author: Jimena Ferreira - FIng- UdelaR
"""
__all__ = ['tree2symb', 'round_expr']

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
        "X11": X11
    }
    final=sympify(ind, locals=locals)
    return final

def round_expr(expr, num_digits):
    from sympy import preorder_traversal,Float, Number
    expr=expr.xreplace({n : Float(n,num_digits ) for n in expr.atoms(Number)})
    return expr

