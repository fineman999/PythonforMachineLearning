# 1.numpy 모듈의 호출
import numpy as np

# 2. array의 생성
test_array = np.array(["1","4",5,7],float)
test_array

type(test_array[3])

test_array = np.array([1,4,5,"8"],np.float32) #String Type의 데이터를 입력해도 통일된다.
test_array

type(test_array[3]) #Float Type으로 자동 형변환을 실시
test_array.dtype  #Array(배열) 전체의 데이터 Type을 반환함

np.array([[1, 4, 5, "8"]], np.float32).shape #행렬 모양

test_array.shape #Array(배열)의 shape을 반환함

# 3. array shape
vector = [1,2,3,4]
np.array(vector, int).shape

matrix = [[1,2,5,8],[1,2,5,8],[1,2,5,8]]
np.array(matrix,int).shape

tensor  = [[[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]],
           [[1,2,5,8],[1,2,5,8],[1,2,5,8]]]
np.array(tensor,int).shape

np.array(tensor,int).ndim #배열의 차원 수

np.array(tensor,int).size #총 개수

# 4. numpy dtype
a = np.array([[1, 2, 3], [4.5, 5, 6]], dtype=int)
np.array([[1,2,3],[4.5,"5","6"]],dtype=np.float32)

np.array([[1,2,3],[4.5,5,6]],dtype=np.float32).nbytes #메모리의 크기를 반환함

np.array([[1, 2, 3], [4.5, "5", "6"]],
         dtype=np.int8).nbytes

np.array([[1, 2, 3], [4.5, "5", "6"]],
         dtype=np.float64).nbytes

# 5. reshape
test_matrix = [[1,2,3,4], [1,2,5,8]]
np.array(test_matrix).shape

np.array(test_matrix).reshape(2,2,2)

np.array(test_matrix).reshape(4,2)


test =np.array(test_matrix).reshape(8,)
test
test.reshape(-1, 1)

np.array(test_matrix).reshape(2,4).shape
np.array(test_matrix).reshape(2,-1).shape #위에 있는 식이랑 같음

# 5. flat or flatten()

test_matrix = [[[1,2,3,4], [1,2,5,8]], [[1,2,3,4], [1,2,5,8]]]
np.array(test_matrix).flatten() #1차원으로 바꿔준다

test_exmaple = np.array([[1, 2, 3], [4.5, 5, 6]], int)
test_exmaple
test_exmaple[0][0]
test_exmaple[0,0]

test_exmaple[0,0] = 10 # Matrix 0,0 에 10 할당
test_exmaple

# 6. slicing
test_exmaple = np.array([
    [1, 2, 5,8], [1, 2, 5,8],[1, 2, 5,8],[1, 2, 5,8]], int)
test_exmaple[:2,:] #row 0~1까지 column은 전체
test_exmaple[:,1:3] #row 전체, column은 1~2까지
test_exmaple[1,:2] #row 1행, column은 까지

test_exmaple = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int)
test_exmaple[:,2:] # 전체 Row의 2열 이상
test_exmaple[1,1:3] # 1 Row의 1열 ~ 2열
test_exmaple[1:3] # 1 Row ~ 2Row의 전체
a = np.arange(100).reshape(10,10)
a[:, -1].reshape(-1,1)

# 7. arrange

np.arange(30).reshape(-1,5) # 6 by 5
np.arange(0, 5, 0.5)
np.arange(30).reshape(5,6) # 5 by 6

# 8. ones, zeros & empty zero는 0으로 초기화됨. empty는  메모리도 초기화되지 않기 때문에 예상하지 못한 쓰레기 값이 들어가 있습니다.
np.zeros(shape=(10,), dtype=np.int8) # 10 - zero vector 생성

np.zeros((2,5)) # 2 by 5 - zero matrix 생성
np.ones(shape=(10,), dtype=np.int8)
np.ones((2,5))

np.empty(shape=(10,), dtype=np.int8)
np.empty((3,5))


test_matrix = np.arange(30).reshape(5,6)
np.zeros_like(test_matrix) #기존의 동일한 모양과 데이터 형태를 유지한 상태에서 반환

# 9. eye, identity & digonal
np.identity(n=3, dtype=np.int8)
np.identity(5)

np.eye(N=3, M=4, dtype=np.int8)
np.eye(5)

np.eye(3,5,k=3)

matrix = np.arange(9).reshape(3,3)
matrix
np.diag(matrix) #대각선에 있는 값들을 추출한다.
np.diag(matrix, k=1)#대각선 위치 지정

"""
균등분포: 확률분포가 취하는 모든 구간에서 일정한 확률을 가지는 분포, 경험적으로 알 수 없을 상황에서 사용하는 것이 균등분포
최소 0, 최대 1, 개수 10개
"""
np.random.uniform(0,1,10).reshape(2,5)

"""
정규분포란  과거의 축적된 경험적 데이타를 이미 보유하고 있어 이를 이용하여 미래에
    발생할 결과값 x의 각 예상되는 범위별로 발생될 확률을 어느정도 추정할 수 있을 때
    사용한다.
    평균 0, 표준편차 1, 개수 10
"""
np.random.normal(0,1,10).reshape(2,5)

"""
 10. operation in array
"""

test_array = np.arange(1,11)
test_array

test_array.sum(dtype=np.float)


test_array = np.arange(1,13).reshape(3,4)
test_array.sum()

test_array.sum(axis=1) # axis=1이므로 두번쨰: 4 각 행을 다 더하면 3개의 원소가 남는다.
test_array.sum(axis=0) # axis=0이므로 첫번쨰: 3 각 열(3)을 다 더하면 4개의 원소가 남는다.

third_order_tensor = np.array([test_array,test_array,test_array])
third_order_tensor.shape

third_order_tensor.sum(axis=2) # shape이 (3,3,4) -> (3,3)
third_order_tensor.sum(axis=1) # shape이 (3,3,4) -> (3,4)
third_order_tensor.sum(axis=0) # shape이 (3,3,4) -> (3,4)

test_array = np.arange(1,13).reshape(3,4)
test_array


test_array.mean(), test_array.mean(axis=0) #mean: 평균
test_array.std(), test_array.std(axis=0) #std: 표준편차
np.exp(test_array), np.sqrt(test_array) #exp: 지수함수, sqrt: 루트

"""
11. Concatenate
"""

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.vstack((a,b)) # 행을 추가

a = np.array([ [1], [2], [3]])
b = np.array([ [2], [3], [4]])
np.hstack((a,b)) #열을 추가

a = np.array([[1, 2, 3]])
b = np.array([[2, 3, 4]])
np.concatenate( (a,b) ,axis=0)
np.concatenate( (a,b) ,axis=1)

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

np.concatenate( (a,b.T) ,axis=1)
a.tolist()

"""
12. Array operations
"""


test_a = np.array([[1,2,3],[4,5,6]], float)
test_a + test_a # Matrix + Matrix 연산

test_a - test_a # Matrix - Matrix 연산
test_a * test_a # Matrix내 element들 간 같은 위치에 있는 값들끼리 연산

matrix_a = np.arange(1,13).reshape(3,4)
matrix_a * matrix_a

"""
13. dot product
"""
test_a = np.arange(1,7).reshape(2,3)
test_b = np.arange(7,13).reshape(3,2)

test_a.dot(test_b) #행렬 곱

test_a = np.arange(1,7).reshape(2,3)
test_a
test_a.transpose()

test_a.T.dot(test_a) # test_a의 transpose x test_a : Matrix 간 곱셈

"""
14. broadcasting
"""
test_matrix = np.array([[1,2,3],[4,5,6]], float)
scalar = 3
test_matrix + scalar # Matrix - Scalar 덧셈
test_matrix - scalar # Matrix - Scalar 뺄셈
test_matrix * 5 # Matrix - Scalar 곱셈
test_matrix / 5 # Matrix - Scalar 나눗셈
test_matrix // 0.2 # Matrix - Scalar 몫
test_matrix ** 2 # Matrix - Scalar 제곱

test_matrix = np.arange(1,13).reshape(4,3)
test_vector = np.arange(10,40,10)
test_matrix+ test_vector
"""
15. numpy performance
"""
def sclar_vector_product(scalar, vector):
    result = []
    for value in vector:
        result.append(scalar * value)
    return result

iternation_max = 1000000

vector = list(range(iternation_max))
scalar = 2
%timeit sclar_vector_product(scalar, vector) # for loop을 이용한 성능

%timeit [scalar * value for value in range(iternation_max)] # list comprehension을 이용한 성능
%timeit np.arange(iternation_max) * scalar # numpy를 이용한 성능

"""
"""
16. all & any
"""
a = np.arange(10)
a
a>5
np.any(a>5), np.any(a<0)
np.all(a>5) , np.all(a < 10)

""""
17. comparision operation¶
""""
test_a = np.array([1, 3, 0], float)
test_b = np.array([5, 2, 1], float)
test_a > test_b
test_a == test_b
(test_a > test_b).any()
a = np.array([1, 3, 0], float)
np.logical_and(a > 0, a < 3) # and 조건의 condition

b = np.array([True, False, True], bool)
np.logical_not(b) # NOT 조건의 condition
a
c = np.array([False, True, False], bool)
np.logical_or(b, c) # OR 조건의 condition
np.where(a > 0, 3, 2) #첫번째 값은 비교, 두번째 값은 조건에 맞을 때 값, 세 번쨰 값은 조건에 맞지 않을 떄 값
np.where(a>0) # 옳을 시 인덱스 값 출력

a = np.arange(5, 15)
a
np.where(a>10)
a = np.array([1, np.NaN, np.Inf], float) #NaN ->Not a Number, Inf -> Infinity
np.isnan(a) #isnan-> not a number인지 아닌지 판별 함수
np.isfinite(a)
#
# """"
# 18. argmax & argmin
# """"
a = np.array([1,2,4,5,8,78,23,3])
np.argmax(a) , np.argmin(a)

a=np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
np.argmax(a, axis=1) , np.argmin(a, axis=0)

""""
19. boolean index
"""""

test_array = np.array([1, 4, 0, 2, 3, 8, 9, 7], float)
test_array > 3
test_array[test_array > 3]

condition = test_array < 3
test_array[condition]


A = np.array([
[12, 13, 14, 12, 16, 14, 11, 10,  9],
[11, 14, 12, 15, 15, 16, 10, 12, 11],
[10, 12, 12, 15, 14, 16, 10, 12, 12],
[ 9, 11, 16, 15, 14, 16, 15, 12, 10],
[12, 11, 16, 14, 10, 12, 16, 12, 13],
[10, 15, 16, 14, 14, 14, 16, 15, 12],
[13, 17, 14, 10, 14, 11, 14, 15, 10],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 17, 19, 16, 17, 18, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 16, 12, 14, 11, 12, 14, 18, 11],
[10, 19, 12, 14, 11, 12, 14, 18, 10],
[14, 22, 12, 14, 11, 12, 14, 17, 13],
[10, 16, 12, 14, 11, 12, 14, 18, 11]])
B = A < 15
B
B.astype(np.int) #데이터 타입을 변환한다. boolean-> int

""""
20. fancy index
"""
a = np.array([2, 4, 6, 8], float)
a[a>4]
a.take(b) #take 함수: bracket index와 같은 효과

a = np.array([[1, 4], [9, 16]], float)
b = np.array([0, 0, 1, 1, 0], int)
c = np.array([0, 1, 1, 1, 1], int)
a[b,c] # b를 row index, c를 column index로 변환하여 표시함
a = np.array([[1, 4], [9, 16]], float)
a[b]
