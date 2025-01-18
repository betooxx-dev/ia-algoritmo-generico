def validate_math_expression(expr):
    allowed_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
    
    test_expr = expr.replace('x', '1')
    test_expr = test_expr.replace('^', '**')
    
    for func in allowed_funcs:
        test_expr = test_expr.replace(func, f'math.{func}')
    
    try:
        result = eval(test_expr)
        return True
    except:
        return False

def compile_function(expr):
    if not expr or not validate_math_expression(expr):
        raise ValueError("Función matemática inválida")
        
    expr = expr.replace('^', '**')
    
    for func in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
        expr = expr.replace(func, f'math.{func}')
    
    try:
        test_func = lambda x: eval(expr)
        test_value = test_func(0)
        return test_func
    except Exception as e:
        raise ValueError(f"Error compiling function: {str(e)}")

def validate_parameters(x_min, x_max, dx, crossover_prob, mutation_prob, mutation_bits_prob, min_population, max_population, generations):
    if x_min >= x_max:
        raise ValueError("El rango mínimo debe ser menor que el máximo")
    if dx <= 0:
        raise ValueError("Delta X debe ser mayor que 0")
    if not 0 <= crossover_prob <= 1:
        raise ValueError("La probabilidad de cruza debe estar entre 0 y 1")
    if not 0 <= mutation_prob <= 1:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= mutation_bits_prob <= 1:
        raise ValueError("La probabilidad de mutación de bits debe estar entre 0 y 1")
    if min_population <= 0:
        raise ValueError("La población mínima debe ser mayor que 0")
    if max_population <= min_population:
        raise ValueError("La población máxima debe ser mayor que la población mínima")
    if generations <= 0:
        raise ValueError("El número de generaciones debe ser mayor que 0")
    return True