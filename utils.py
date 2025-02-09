import sympy

def symbolic_conversion(function_string, x_data):
    x_sym = sympy.Symbol('x', real=True)
    expr = sympy.sympify(function_string)
    func = sympy.lambdify(x_sym, expr, 'numpy')
    y_data = func(x_data)
    return y_data
