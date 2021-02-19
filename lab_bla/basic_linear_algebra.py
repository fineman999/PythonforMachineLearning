def vector_size_check(*vector_variables):
    return all(len(vector_variables[0]) == x
                for x in [len(vector) for vector in vector_variables[1:]])
vector_size_check([1,2,3],[2,3,4],[3,5,6])

def vector_addition(*vector_variables):
    if(vector_size_check(*vector_variables)==True):
        return [sum(t) for t in zip(*vector_variables)]
    else:
        return "Error"
print(vector_addition([1, 5], [10, 4], [4, 7]))
print(vector_addition([1, 3, 4], [4], [6,7]))


def vector_subtraction(*vector_variables):
    if vector_size_check(*vector_variables) == False:
        raise ArithmeticError #예외처리
    return [t[0]-sum(t[1:]) for t in zip(*vector_variables)]

        print(vector_subtraction([1, 3], [2, 4]))
        print(vector_subtraction([1, 5], [10, 4], [4, 7]))


def scalar_vector_product(alpha, vector_variable):
    return [alpha*t for t in vector_variable]

print (scalar_vector_product(5,[1,2,3]))
print (scalar_vector_product(3,[2,2])) # Expected value: [6, 6]
print (scalar_vector_product(4,[1])) # Expected value: [4]

def matrix_size_check(*matrix_variables):
    return (all([len(set(len(matrix[0]) for matrix in matrix_variables)) == 1]) and
    all([len(matrix_variables[0]) == len(matrix) for matrix in matrix_variables]))


matrix_x = [[2, 2], [2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]
matrix_w = [[2, 5], [1, 1], [2, 2]]

matrix_x[0]
print (matrix_size_check(matrix_x, matrix_y, matrix_z)) # Expected value: False
print (matrix_size_check(matrix_y, matrix_z)) # Expected value: True
print (matrix_size_check(matrix_x, matrix_w)) # Expected value: # Expected value: True

def is_matrix_equal(*matrix_variables):
    return  all([all([i[0]==sum(i)/len(i) for i in zip(*matrix)]) for matrix in zip(*matrix_variables)])

matrix_x = [[1, 2], [3, 4]]
matrix_y = [[5, 6], [7, 8]]
print (is_matrix_equal(matrix_x, matrix_y, matrix_y, matrix_y)) # Expected value: False
print (is_matrix_equal(matrix_x, matrix_x)) # Expected value: True


def matrix_addition(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    return  [([sum(i) for i in zip(*matrix)]) for matrix in zip(*matrix_variables)]

# 실행결과
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]

print (matrix_addition(matrix_x, matrix_y)) # Expected value: [[4, 7], [4, 3]]
print (matrix_addition(matrix_x, matrix_y, matrix_z)) # Expected value: [[6, 11], [9, 6]]

def matrix_subtraction(*matrix_variables):
    if matrix_size_check(*matrix_variables) == False:
        raise ArithmeticError
    return [([i[0]-sum(i[1:])  for i in zip(*matrix)]) for matrix in zip(*matrix_variables)]

# 실행결과
matrix_x = [[2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]

print (matrix_subtraction(matrix_x, matrix_y)) # Expected value: [[0, -3], [0, 1]]
print (matrix_subtraction(matrix_x, matrix_y, matrix_z)) # Expected value: [[-2, -7], [-5, -2]]

def matrix_transpose(matrix_variable):
    return [[*t] for t in zip(*matrix_variable)]

# 실행결과
matrix_w = [[2, 5], [1, 1], [2, 2]]
matrix_transpose(matrix_w)

def scalar_matrix_product(alpha, matrix_variable):
    return [[alpha*t for t in vector_variable] for vector_variable in matrix_variable]

# 실행결과
matrix_x = [[2, 2], [2, 2], [2, 2]]
matrix_y = [[2, 5], [2, 1]]
matrix_z = [[2, 4], [5, 3]]
matrix_w = [[2, 5], [1, 1], [2, 2]]

print(scalar_matrix_product(3, matrix_x)) #Expected value: [[6, 6], [6, 6], [6, 6]]
print(scalar_matrix_product(2, matrix_y)) #Expected value: [[4, 10], [4, 2]]
print(scalar_matrix_product(4, matrix_z)) #Expected value: [[8, 16], [20, 12]]
print(scalar_matrix_product(3, matrix_w)) #Expected value: [[6, 15], [3, 3], [6, 6]]


def is_product_availability_matrix(matrix_a, matrix_b):
    return len([column_vector for column_vector in zip(*matrix_a)]) == len(matrix_b)

# 실행결과
matrix_x= [[2, 5], [1, 1]]
matrix_y = [[1, 1, 2], [2, 1, 1]]
matrix_z = [[2, 4], [5, 3], [1, 3]]
matrix_l = [[1,1]]
print(is_product_availability_matrix(matrix_y, matrix_z)) # Expected value: True
print(is_product_availability_matrix(matrix_z, matrix_x)) # Expected value: True
print(is_product_availability_matrix(matrix_z, matrix_l)) # Expected value: False //matrix_w가없습니다
print(is_product_availability_matrix(matrix_x, matrix_x)) # Expected value: True

def matrix_product(matrix_a, matrix_b):
    if is_product_availability_matrix(matrix_a, matrix_b) == False:
        return False
    return [[sum(a*b for a, b in zip(row_a, column_b))  for column_b in zip(*matrix_b)] for row_a in matrix_a]

# 실행결과
matrix_x= [[2, 5], [1, 1]]
matrix_y = [[1, 1, 2], [2, 1, 1]]
matrix_z = [[2, 4], [5, 3], [1, 3]]

print(matrix_product(matrix_y, matrix_z)) # Expected value: [[9, 13], [10, 14]]
print(matrix_product(matrix_z, matrix_x)) # Expected value: [[8, 14], [13, 28], [5, 8]]
print(matrix_product(matrix_x, matrix_x)) # Expected value: [[9, 15], [3, 6]]
print(matrix_product(matrix_z, matrix_l)) # Expected value: False