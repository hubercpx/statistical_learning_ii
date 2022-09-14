import pandas as pd
import numpy as np
import re
from sympy import diff, solveset, Eq, Interval, S
from sympy.abc import x,y
from math import *
import os.path
# import os 


p = re.compile('[\+|\-|*|/]')
math_operator = ["+", "-", "*", "/"]
lab4_prefix_filename = "datalab4"

def process_exp(func_str):
  e_index = func_str.find("e^")
  if (e_index >= 0):
    padd = 2
    if ( not(func_str[e_index + padd] == "(") ):
      k = p.search(func_str[e_index+padd:])
      if (k):
        i = e_index + padd + k.start()
        func_str = func_str[:i] + ")" + func_str[i:]
        func_str = func_str[:e_index] + "exp(" + func_str[e_index+padd:]
      else: #ya no encontro signo
        func_str = func_str[:e_index] + "exp(" + func_str[e_index+padd:] + ")"
    else:
      func_str = func_str[:e_index] + "exp" + func_str[e_index+padd:]
  if (func_str.find("e^") >= 0):
    func_str = process_exp(func_str)
    
  return func_str


def add_one_before_x(func_str):
  result_func = ""
  for i in range(len(func_str)):
    one = "1" if (func_str[i] == "x" and (i==0 or func_str[i-1] in math_operator) ) else ""
    result_func += one + func_str[i]
  return result_func


def replace_x(func_str):
  result_func = ""
  for i in range(len(func_str)):
    result_func += "*(x)" if (func_str[i] == "x" and func_str[i-1].isnumeric()) else func_str[i]
  return result_func


def transform_function(str_equ):
  str_equ = process_exp(str_equ)
  str_equ = add_one_before_x(str_equ)
  str_equ = replace_x(str_equ)

  # .replace("x", '*(x)')\
  strOut = str_equ\
          .replace("^", "**")
  # print("transform_function:", strOut)
  return strOut

  
#Evaluación REGREX
def evaluate_Fx(str_equ, valX):
  x = valX
  strOut = transform_function(str_equ)
  out = eval(strOut)
  #print("evaluate_Fx:::", strOut)
  return out

#Evaluación REGREX
def evaluate_derivate_Fx(str_equ, valX):
  x = valX
  strOut = transform_function(str_equ)
  strOut = derive_function(strOut)
  out = eval(strOut)
  #print("evaluate_derive_Fx:::", strOut)
  return out

def derive_function(str_equ):
  pattern1 = re.compile('\(x\)\*\*\d')
  pattern2 = re.compile('x\*\*\d')
  vars = pattern1.findall(str_equ) + pattern2.findall(str_equ)
  
  dictMap = {}
  for var in vars:
    out = var[-1] + '*' + var[0:-1] + '('+var[-1]+'-1)' 
    dictMap[var] = out
    
  for key, value in dictMap.items():
    str_equ = str_equ.replace(key, value)
  
  return str_equ
  
def derivar(str_equ, x):
  #dfx/dx
  strFx = str_equ.replace("x", '*(y)')
  strFx = strFx.replace("^", "**")
  dx = diff(strFx,y).subs(y,x) 
  return dx


#Deferencias finitas para derivadas
def evaluate_derivate_fx1(str_equ, x, h):
  x = float(x)
  h = float(h)
  
  dx = derivar(str_equ,x)
  
  #f(x0+h)
  strOut = str_equ.replace("x", '*(x + h)')
  strOut = strOut.replace("^", "**")
  out = eval(strOut)
  
  #f(x0-h)
  strOut = str_equ.replace("x", '*(x - h)')
  strOut = strOut.replace("^", "**")
  out = out - eval(strOut)
  
  #f(x0+h) - f(x0-h) / 2
  out = out/(2*h)
  
  datos = {'ValorReal':[float(dx)],
          'Aproximación':[out],
          'Error': [float(abs(dx-out))]}
  
  return pd.DataFrame(datos)

def evaluate_derivate_fx2(str_equ, x, h):
  x = float(x)
  h = float(h)
  
  dx = derivar(str_equ,x)
  
  strOut = str_equ.replace("x", '*(x + h)')
  strOut = strOut.replace("^", "**")
  strOut = "-4*(" + strOut + ")"
  out = eval(strOut)
  
  strOut = str_equ.replace("x", '*(x + 2*h)')
  strOut = strOut.replace("^", "**")
  out = out + eval(strOut)
  
  strOut = str_equ.replace("x", '*(x)')
  strOut = strOut.replace("^", "**")
  strOut = "3*(" + strOut + ")"
  out = out + eval(strOut)
  
  out = -out/(2*h)
  
  datos = {'ValorReal':[float(dx)],
          'Aproximación':[out],
          'Error': [float(abs(dx-out))]}
  
  return pd.DataFrame(datos)

def evaluate_derivate_fx3(str_equ, x, h):
  x = float(x)
  h = float(h)
  
  dx = derivar(str_equ,x)
  
  strOut = str_equ.replace("x", '*(x - 2*h)')
  strOut = strOut.replace("^", "**")
  out = eval(strOut)
  
  strOut = str_equ.replace("x", '*(x - h)')
  strOut = strOut.replace("^", "**")
  strOut = "8*(" + strOut + ")"
  out = out - eval(strOut)
  
  strOut = str_equ.replace("x", '*(x + h)')
  strOut = strOut.replace("^", "**")
  strOut = "8*(" + strOut + ")"
  out = out + eval(strOut)
  
  strOut = str_equ.replace("x", '*(x + 2*h)')
  strOut = strOut.replace("^", "**")
  out = out - eval(strOut)
  
  out = out/(12*h)
  datos = {'ValorReal':[float(dx)],
          'Aproximación':[out],
          'Error': [float(abs(dx-out))]}
  
  return pd.DataFrame(datos)


#Deferencias finitas para derivadas
def evaluate_derivate_fx(str_equ, x, h):
  x = float(x)
  h = float(h)
  
  #f(x0+h)
  strOut = str_equ.replace("x", '*(x + h)')
  strOut = strOut.replace("^", "**")
  out = eval(strOut)
  
  #f(x0-h)
  strOut = str_equ.replace("x", '*(x - h)')
  strOut = strOut.replace("^", "**")
  out = out - eval(strOut)
  
  #f(x0+h) - f(x0-h) / 2
  out = out/(2*h)
  
  return out

#Resolverdor de Newton
def newtonSolverX(x0, f_x, eps):
  x0 = float(x0)
  eps = float(eps)
  xn = x0
  error = 1
  arrayIters = []
  arrayF_x = []
  arrayf_x = []
  arrayXn = []
  arrayErr = []
  
  i = 0
  h = 0.000001
  while(error > eps):
    print("...")
    x_n1 = xn - (evaluate_Fx(f_x, xn)/evaluate_derivate_fx(f_x, xn, h))
    error = abs(x_n1 - xn)
    i = i + 1
    xn = x_n1
    arrayIters.append(i)
    arrayXn.append(xn)
    arrayErr.append(error)
    solution = [i, xn, error]

  print("Finalizo...")
  TableOut = pd.DataFrame({'Iter':arrayIters, 'Xn':arrayXn, 'Error': arrayErr})
  return TableOut

def add(a, b):
  a = int(a)
  b = int(b)
  resultado = a + b
  return "El resultado es: " + str(resultado)


#Deferencias finitas para derivadas
def evaluate_derivate_fx1XY(str_equ, x, y, h):
  x = float(x)
  y = float(y)
  h = float(h)
  
  #f(x0+h,y)
  ecuacionX1 = str_equ
  ecuacionX1 = ecuacionX1.replace("x", '*(x + h)')
  ecuacionX1 = ecuacionX1.replace("y", '*(y)')
  ecuacionX1 = ecuacionX1.replace("^", "**")
  outX = eval(ecuacionX1)
  
  #f(x0-h,y)
  ecuacionX2 = str_equ
  ecuacionX2 = ecuacionX2.replace("x", '*(x - h)')
  ecuacionX2 = ecuacionX2.replace("y", '*(y)')
  ecuacionX2 = ecuacionX2.replace("^", "**")
  outX = outX - eval(ecuacionX2)
  
  #f(x0+h,y) - f(x0-h,y) / 2h
  outX = outX/(2*h)
  
  #f(x0,y0+h)
  ecuacionY1 = str_equ
  ecuacionY1 = ecuacionY1.replace("y", '*(y + h)')
  ecuacionY1 = ecuacionY1.replace("x", '*(x)')
  ecuacionY1 = ecuacionY1.replace("^", "**")
  outY = eval(ecuacionY1)
  
  #f(x0,y0-h)
  ecuacionY2 = str_equ
  ecuacionY2 = ecuacionY2.replace("y", '*(y - h)')
  ecuacionY2 = ecuacionY2.replace("x", '*(x)')
  ecuacionY2 = ecuacionY2.replace("^", "**")
  outY = outY - eval(ecuacionY2)
  
  #f(x0,y0+h) - f(x0,y0-h) / 2h
  outY = outY/(2*h)
  
  datos = {'Aprox df/dx':[outX],
          'Aprox df/dy':[outY]}
  
  return pd.DataFrame(datos)


def evaluate_derivate_fx2XY(str_equ, x, y, h):
  x = float(x)
  y = float(y)
  h = float(h)
  
  #dx = derivar(str_equ,x)
  dx=0
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x + h)')
  ecuacionX = ecuacionX.replace("y", '*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  ecuacionX = "-4*(" + ecuacionX + ")"
  outX = eval(ecuacionX)
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x + 2*h)')
  ecuacionX = ecuacionX.replace("y", '*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  outX = outX + eval(ecuacionX)
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x)')
  ecuacionX = ecuacionX.replace("y", '*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  ecuacionX = "3*(" + ecuacionX + ")"
  outX = outX + eval(ecuacionX)
  
  outX = -outX/(2*h)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("y", '*(y + h)')
  ecuacionY = ecuacionY.replace("x", '*(x)')
  ecuacionY = ecuacionY.replace("^", "**")
  ecuacionY = "-4*(" + ecuacionY + ")"
  outY = eval(ecuacionY)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("y", '*(y + 2*h)')
  ecuacionY = ecuacionY.replace("x", '*(x)')
  ecuacionY = ecuacionY.replace("^", "**")
  outY = outY + eval(ecuacionY)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("x", '*(x)')
  ecuacionY = ecuacionY.replace("y", '*(y)')
  ecuacionY = ecuacionY.replace("^", "**")
  ecuacionY = "3*(" + ecuacionY + ")"
  outY = outY + eval(ecuacionY)
  
  outY = -outY/(2*h)
  
  datos = {'ValorReal df/dx':[float(dx)],
           'ValorReal df/dy':[float(dx)],
          'Aprox df/dx':[outX],
          'Aprox df/dy':[outY],
          'Norma': [float(np.sqrt(outX**2+outY**2))]}
  
  return pd.DataFrame(datos)


def evaluate_derivate_fx3XY(str_equ, x, y, h):
  x = float(x)
  y = float(y)
  h = float(h)
  
  #dx = derivar(str_equ,x)
  dx=0
  
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x - 2*h)')
  ecuacionX = ecuacionX.replace("y",'*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  outX = eval(ecuacionX)
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x - h)')
  ecuacionX = ecuacionX.replace("y",'*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  ecuacionX = "8*(" + ecuacionX + ")"
  outX = outX - eval(ecuacionX)
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x + h)')
  ecuacionX = ecuacionX.replace("y",'*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  ecuacionX = "8*(" + ecuacionX + ")"
  outX = outX + eval(ecuacionX)
  
  ecuacionX = str_equ
  ecuacionX = ecuacionX.replace("x", '*(x + 2*h)')
  ecuacionX = ecuacionX.replace("y",'*(y)')
  ecuacionX = ecuacionX.replace("^", "**")
  outX = outX - eval(ecuacionX)
  
  outX = outX/(12*h)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("y", '*(y - 2*h)')
  ecuacionY = ecuacionY.replace("x",'*(x)')
  ecuacionY = ecuacionY.replace("^", "**")
  outY = eval(ecuacionY)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("y", '*(y - h)')
  ecuacionY = ecuacionY.replace("x",'*(x)')
  ecuacionY = ecuacionY.replace("^", "**")
  ecuacionY = "8*(" + ecuacionY + ")"
  outY = outY - eval(ecuacionY)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("y", '*(y + h)')
  ecuacionY = ecuacionY.replace("x",'*(x)')
  ecuacionY = ecuacionY.replace("^", "**")
  ecuacionY = "8*(" + ecuacionY + ")"
  outY = outY + eval(ecuacionY)
  
  ecuacionY = str_equ
  ecuacionY = ecuacionY.replace("y", '*(y + 2*h)')
  ecuacionY = ecuacionY.replace("x",'*(x)')
  ecuacionY = ecuacionY.replace("^", "**")
  outY = outY - eval(ecuacionY)
  
  outY = outY/(12*h)
  
  datos = {'ValorReal df/dx':[float(dx)],
           'ValorReal df/dy':[float(dx)],
          'Aprox df/dx':[outX],
          'Aprox df/dy':[outY],
          'Norma': [float(np.sqrt(outX**2+outY**2))]}
  
  return pd.DataFrame(datos)


"""
Ejecuta y evalua el metodo de biseccion.
PARAMETROS:
  f_x, 
  a, 
  b, 
  kmax, 
  tolerance
Retorna un pandas.dataframe con los valores de las iteraciones.
"""
def evaluate_bisection(f_x, a, b, kmax, tolerance):
  a = float(a)
  b = float(b)
  f_a = float(evaluate_Fx(f_x, a))
  f_b = float(evaluate_Fx(f_x, b))
  dict_result= {'Iteracion':[],
                'Xk':[],
                'Error':[]}
  if ((f_a*f_b) > 0):
    return pd.DataFrame.from_dict(dict_result)
  kmax = float(kmax)
  tolerance = float(tolerance)
  k = 0
  xk = (a+b)/2
  f_xk = evaluate_Fx(f_x, xk)
  # transformed_fx = transform_function(f_x) 
  # equation = Eq(*S(transformed_fx+", 0"))
  # rootss = solveset(equation, x, Interval(a,b))
  # real_x = eval(str( tuple(rootss)[0] ))
  while k < kmax and abs(f_xk) > tolerance:
    f_a = evaluate_Fx(f_x, a)
    # print("k:",k, "  a:",a, "\tb",b, "\txk:",xk, "\tf_xk:",f_xk, "\tf_a:",f_a)
    if (f_a*f_xk < 0):
      b = xk
    else:
      a = xk
    k += 1
    dict_result["Iteracion"].append(int(k))
    dict_result["Xk"].append(float(xk))
    dict_result["Error"].append(float(abs(f_xk)))
    xk = (a+b)/2
    f_xk = evaluate_Fx(f_x, xk)

  return pd.DataFrame.from_dict(dict_result)

"""
Ejecuta y evalua el metodo de Newton Raphson.
PARAMETROS:
  f_x, 
  x, 
  kmax, 
  tolerance
Retorna un pandas.dataframe con los valores de las iteraciones.
"""
def evaluate_NR(f_x, x, kmax, tolerance):
  x = float(x)
  kmax = float(kmax)
  tolerance = float(tolerance)
  F_x = float(evaluate_Fx(f_x, x))
  k = 0
  
  dict_result= {'Iteracion':[],
                'Xk':[],
                'Error':[]}
  
  while k < kmax and abs(F_x) > tolerance:
    x -= evaluate_Fx(f_x, x)/evaluate_derivate_Fx(f_x, x)
    k += 1
    
    dict_result["Iteracion"].append(int(k))
    dict_result["Xk"].append(float(x))
    dict_result["Error"].append(float(abs(F_x)))
    
    F_x = float(evaluate_Fx(f_x, x))
  
  return pd.DataFrame.from_dict(dict_result)


"""
"""
def evaluate_rosenbrock(xo, alpha, epsilon, kmax):
  dict_result= {'k':[],
                'Xk':[],
                'Pk':[],
                '||GDf(Xk)||':[]}
  k = 0
  alpha = float(alpha)
  # kmax = 1000
  epsilon = float(epsilon)
  Q = np.array([[8,4],[4,4]])
  df_dx = "8*x+4*w-3"  #x1
  df_dw = "4*w+4*x"    #x2
  df_dx = "-400*x*(w-x**2)-2*(1-x)"  #x1
  df_dw = "200*(w-x**2)"    #x2
  xo = np.matrix(xo, dtype=float)
  xk = np.transpose(xo)
  x = xk[0, 0]
  w = xk[1, 0]
  gd_f = np.array([[eval(df_dx)], [eval(df_dw)]])
  gd_f_norm = np.linalg.norm(np.power(gd_f, 2), 1)
  # print("k=", k, "; abs(gd_f_norm)=", abs(gd_f_norm))

  while abs(gd_f_norm) > epsilon and k < kmax:
    # para calcular alpha a usar
    # gd_f_norm = np.linalg.norm(np.power(gd_f, 2), 1)
    # gd_f_T_by_q = np.transpose(gd_f)*Q
    
    dict_result['k'].append(str(k))
    dict_result['Xk'].append("[{:.8f}, {:.8f}]".format(x, w))
    dict_result['Pk'].append("[{:.8f}, {:.8f}]".format(gd_f[0,0], gd_f[1,0]))
    dict_result['||GDf(Xk)||'].append(float(gd_f_norm))
    
    xk_plus1 = xk - alpha * gd_f
    xk = xk_plus1
    x = xk[0, 0]
    w = xk[1, 0]
    gd_f = np.array([[eval(df_dx)], [eval(df_dw)]])
    gd_f_norm = np.linalg.norm(np.power(gd_f, 2), 1)
    k += 1

  return pd.DataFrame.from_dict(dict_result)



def evaluate_gdvariant(variant, epsilon, kmax, learning_rates, epochs, batch_sizes, path):
  variant = int(variant)
  epsilon = float(epsilon)
  learning_rates = eval(learning_rates)
  batch_sizes = eval(batch_sizes)
  # path = os.path.dirname(os.path.realpath(__file__))
  # path = path.replace("\\", "\\\\") + "\\\\"
  if (not os.path.exists(path + lab4_prefix_filename+'_A.csv')):
    create_persist_data(parent_path = path)
  A = np.genfromtxt(path + lab4_prefix_filename + '_A.csv', delimiter=',', dtype=None)
  b = np.genfromtxt(path + lab4_prefix_filename + '_b.csv', delimiter=',', dtype=None)
  b = b.reshape((b.shape[0], 1))
  x_true = np.genfromtxt(path + lab4_prefix_filename + '_x_true.csv', delimiter=',', dtype=None)
  x_true = x_true.reshape((x_true.shape[0], 1))
  n = A.shape[0]
  d = A.shape[1]
  dict_result= {}
  fx = 0

  if (variant == 1):
    AT = A.transpose()
    ATdotA = AT.dot(A)
    ATdotA_inv = np.linalg.inv(ATdotA)
    ATdotB = AT.dot(b)
    x_asterisk = ATdotA_inv.dot(ATdotB)
    # print("SHAPES::: A:", A.shape, "; b:", b.shape, "; AT:", AT.shape, "; ATdotA:", ATdotA.shape, "; x_asterisk:", x_asterisk.shape)
    fx = compute_LSS(A, b, x_asterisk)
    dict_result = {'x*': x_asterisk.reshape((1, x_asterisk.shape[0]))[0].tolist(), 
                  "f(x)": [fx for i in range(x_asterisk.shape[0])]
                  }

  elif (variant == 2):  #GD
    dict_result["i_th"] = []
    k_lr = {}
    k = 0
    while k < kmax:
      dict_result["i_th"].append(k)
      for alpha in learning_rates:
        if ("fi_"+str(alpha) not in dict_result):
          dict_result["fi_"+str(alpha)] = []
        if (alpha in k_lr):
          xk = k_lr[alpha]
        else:
          xk = np.zeros((d, 1))
        # for epoch in range(epochs):
        gradient = get_gradient_by_sum(A, b, xk)
        xk1 = xk - alpha * gradient
        xk = xk1
        k_lr[alpha] = xk
        fx = compute_LSS(A, b, xk)
        dict_result["fi_"+str(alpha)].append(float(fx))
      k += 1

  elif (variant == 3):  #SGD
    dict_result["i_th"] = []
    k_lr = {}
    k = 0
    while k < kmax:
      dict_result["i_th"].append(k)
      for alpha in learning_rates:
        if ("fi_"+str(alpha) not in dict_result):
          dict_result["fi_"+str(alpha)] = []
        if (alpha in k_lr):
          xk = k_lr[alpha]
        else:
          xk = np.zeros((d, 1))
        for epoch in range(epochs):
          rndi = np.random.randint(0, n, size=(1))[0] #indice aleatorio de la observacion
          gradient = get_gradient_by_sum(A[rndi, :].reshape((1, d)), 
                                        b[rndi, :].reshape((1, 1)), 
                                        xk)
          xk1 = xk - alpha * gradient
          xk = xk1
        k_lr[alpha] = xk
        fx = compute_LSS(A, b, xk)
        dict_result["fi_"+str(alpha)].append(float(fx))
      k += 1

  elif (variant == 4):  #MBGD
    dict_result["i_th"] = []
    k_lr = {}
    k = 0
    while k < kmax:
      dict_result["i_th"].append(k)
      for bsize in batch_sizes:
        if (A.shape[0] >= bsize):      
          for alpha in learning_rates:
            colname = "fi_"+str(bsize)+"_"+str(alpha)
            if (colname not in dict_result):
              dict_result[colname] = []
            if ((bsize+alpha) in k_lr):
              xk = k_lr[bsize+alpha]
            else:
              xk = np.zeros((d, 1))
            for epoch in range(epochs):
              gradient = get_gradient_by_sum(A[:bsize, :], 
                                          b[:bsize, :].reshape((bsize, 1)),
                                          xk)
              xk1 = xk - alpha * gradient
              xk = xk1
            k_lr[alpha] = xk
            fx = compute_LSS(A, b, xk)
            dict_result[colname].append(float(fx))
          k += 1

    # dict_result = {"i_th": [epoch+1 for epoch in range(epochs)]}
    # for alpha in learning_rates:
    #   xk = np.zeros((d, 1)) #valores iniciales, random
    #   for bsize in batch_sizes:
    #     if (A.shape[0] >= bsize):
    #       fi_ai = []
    #       for epoch in range(epochs):
    #         fx = compute_LSS(A, b, xk)
    #         fi_ai.append(float(fx))
    #         gradient = get_gradient_by_sum(A[:bsize, :], 
    #                                       b[:bsize, :].reshape((bsize, 1)), 
    #                                       xk)
    #         xk1 = xk - alpha * gradient
    #         xk = xk1
    #       dict_result["fi_"+str(bsize)+"_"+str(alpha)] = fi_ai

  print("\tPROCESO TERMINADO")
  return pd.DataFrame.from_dict(dict_result)



def compute_LSS(A, b, x_asterisk):
  fx = sum([(x_asterisk.transpose().dot(A[i]) - b[i])**2 \
            for i in range(A.shape[0])])
  return fx



"""
  Ejecuta la funcion de costo Least Square Sum (LSS)
  Parametros
    A:
    b:
    Xk:
"""
def get_gradient_by_sum(A, b, Xk):
  n = A.shape[0]
  d = A.shape[1]
  gradient = np.zeros_like(Xk)
  for i in range(n):
    ai  = A[i, :].reshape((1, d))
    aiT = A[i, :].transpose()
    gradient = gradient + (aiT.dot(Xk) - b[i, :]).dot(ai)
  # (1/n) * SUM((aiT*xk -bi)*ai)
  gradient = gradient.sum(axis = 1).reshape((d, 1))
  gradient = (1/n) * gradient
  return gradient


"""
  d: cantidad de columnas para el dataset.
  n: cantidad de observaciones para el dataset.
  parent_path: ubicacion de los archivos de datos
"""
def create_persist_data(n = 1000, d = 100, parent_path = ""):
  A = np.random.normal(0, 1, size=(n, d))
  x_true = np.random.normal(0, 1, size=(d, 1))
  b = A.dot(x_true) + np.random.normal(0, 0.5, size=(n, 1))
  path_filename = parent_path + lab4_prefix_filename
  np.savetxt(path_filename + '_A.csv', A, delimiter = ',')
  np.savetxt(path_filename + '_x_true.csv', x_true, delimiter = ',')
  np.savetxt(path_filename + '_b.csv', b, delimiter = ',')
  

# def main():
#   variant = 3
#   epsilon = 0.00001
#   kmax = 1000
#   learning_rates = "[0.0005, 0.005, 0.01]"
#   epochs = 1000
#   batch_sizes = "[25,50,200]"
#   path = "C:\\Users\\hbarrientosg\\Documents\\Galileo\\2021T03\\Algoritmos Ciencia Datos\\Laboratorios\\LaboratoriosACD\\"
#   result = evaluate_gdvariant(variant, epsilon, kmax, learning_rates, epochs, batch_sizes, path)
# 
# if __name__ == "__main__":
#     main()
