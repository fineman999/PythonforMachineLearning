import pandas as pd

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data' #Data URL
# data_url = './housing.data' #Data URL
df_data = pd.read_csv(data_url, sep='\s+', header = None) #csv 타입 데이터 로드, separate는 빈공간으로 지정하고, Column은 없음

df_data.head() # head: 처음 다섯줄 출력


df_data.columns = [
    'CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO' ,'B', 'LSTAT', 'MEDV']
# Column Header 이름 지정
df_data.head()
type(df_data.values) #타입은 numpy

"""
 1. pandas_series
"""
from pandas import Series, DataFrame
import numpy as np

list_data = [1,2,3,4,5]
example_obj = Series(data = list_data)
example_obj

list_data = [1,2,3,4,5]
list_name = ["a","b","c","d","e"]
example_obj = Series(data = list_data, index=list_name)
example_obj
example_obj.index
example_obj.values

type(example_obj.values)

dict_data = {"a":1, "b":2, "c":3, "d":4, "e":5}
example_obj = Series(dict_data, dtype=np.float32, name="example_data")
example_obj
example_obj["a"]

example_obj["a"] = 3.2
example_obj

example_obj[example_obj > 2]
example_obj * 2

np.exp(example_obj) #np.abs , np.log 지수함수로 만들기 e**0
"b" in example_obj
example_obj.to_dict() #from DataFrame to dictionary
example_obj.values
example_obj.index
example_obj.name = "number"
example_obj.index.name = "alphabet"
example_obj

dict_data_1 = {"a":1, "b":2, "c":3, "d":4, "e":5}
indexes = ["a","b","c","d","e","f","g","h"]
series_obj_1 = Series(dict_data_1, index=indexes)
series_obj_1
"""
 2. pandas_dataframe
"""
# Example from - https://chrisalbon.com/python/pandas_map_values_to_values.html
raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
        'age': [42, 52, 36, 24, 73],
        'city': ['San Francisco', 'Baltimore', 'Miami', 'Douglas', 'Boston']}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'city'])
df
DataFrame(raw_data, columns = ["age", "city"])
DataFrame(raw_data,
          columns = ["first_name","last_name","age", "city", "debt"]
         )

df = DataFrame(raw_data, columns = ["first_name","last_name","age", "city", "debt"])
df.first_name
df["first_name"]
df
df.loc[1] #loc: '변수명'을 기준으로 데이터프레임을 분리
df["age"].iloc[1:] #'인덱스 번호'를 기준으로 데이터프레임을 분리
df.iloc[1]
# Example from - https://stackoverflow.com/questions/31593201/pandas-iloc-vs-ix-vs-loc-explanation
s = pd.Series(np.nan, index=[49,48,47,46,45, 1, 2, 3, 4, 5]) #nan means not a num
s
s.iloc[:3]
df.age > 40
df.debt = df.age > 40
df
values = Series(data=["M","F","F"],index=[0,1,3])
values
df["sex"] = values
df
df.T
df.values
df.to_csv()
del df["debt"]
df
# Example from Python for data analyis

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

DataFrame(pop)
"""
3. data_selection
"""
#
# Data loading
# xlrd 모듈이 없을 경우, conda install xlrd

!conda install --y xlrd
df = pd.read_excel("./winemag-data-130k-v2.csv")
df = pd.read_csv("./winemag-data-130k-v2.csv") #csv 타입 데이터 로드, separate는 빈공간으로 지정하고, Column은 없음
df.head()
# data selction with index number and column names
df["country"].head(2)
df[["country","description","designation"]].head(3)
df[:10]
df["country"][:3]
country_serires = df["country"]
country_serires[:3]
country_serires[[1,5,2]]

price_serires = df["price"]
price_serires[price_serires<15]
df.index = df["price"]
del df["price"]
df.head()

df[["country","description"]]
df.loc[[8,9],["country","description"]]

df[["country", "description"]].iloc[:10]
# reindex
df.index = list(range(0,129971 ))
df.head()
df
df.drop(1)
df

matrix = df.values
matrix[:3]
matrix[:,-3:]
matrix[:,-3:].sum(axis=1)
df.drop("country",axis=1).head()
df.drop(["country", "description"],axis=1)
matrix = df.values
matrix
"""
4. map_apply_lambda
 lambda: 이름이 없는 함수, 익명함수
 lambda 매개변수 : 표현식
 def add(n,m):
    return n+m
print(add(2,3))         ->
                                print((lambda n,m:n+m)(2,3))
lambdaAdd = lambda n,m:n+m
print(lambdaAdd(2,3))
#5
print(lambdaAdd(4,5))
#9

map(함수, 리스트)

reduce(함수, 순서형 자료)

reduce(lambda x, y: x + y, [0, 1, 2, 3, 4])
10
 """
# lambda
(lambda x: x +1)(5)

# map & replace

ex = [1,2,3,4,5]
ey = [2,3,4,5,6]
f = lambda x: x ** 2
list(map(f, ex))

f = lambda x, y: x + y
list(map(f, ex, ey))

list(map(lambda x: x+5, ex))
#python 3에는 list를 꼭 붙여줘야함

s1 = Series(np.arange(10))
s1.head(5)
s1.map(lambda x: x**2).head(5)
s1
z = {1: 'A', 2: 'B', 3: 'C'}
s1.map(z)
s1
s2 = Series(np.arange(10,20))
s2
s1.map(s2)

df=pd.read_csv("https://raw.githubusercontent.com/rstudio/Intro/master/data/wages.csv")
df.head()
df.sex.unique() #유일한 값만 찾게 나옴
df["sex_code"] =  df.sex.map({"male":0, "female":1})
df.head(5)
df.sex.replace(
    {"male":0, "female":1}
).head()
df.sex.head(5)


# inplace의 장점은 100개의 데이터가 있다고 할 때 100개의 변수를 선언하는 대신 하나의 변수를 재활용함으로써 메모리 사용률을 극도로 아낄 수 있다.
 # 단, 단점으로 코드의 가독성이 매우 떨어지게 된다.
df.sex.replace(
    ["male", "female"],
    [0,1], inplace=True)
df

del df["sex_code"]

df.head()

# apply & applymap
df_info = df[["earn", "height","age"]]
df_info.head()
f = lambda x : x.max() - x.min()
df_info.apply(f)
df_info.apply(sum)
df_info.sum()


def f(x):
    return Series([x.min(), x.max(), x.mean()],
                    index=["min", "max", "mean"])
df_info.apply(f)
f = lambda x : -x
df_info.applymap(f).head(5)

f = lambda x : -x
df_info["earn"].apply(f).head(5)
"""
 apply는 DataFrame의 행 / 열 기준으로 작동하고 applymap는 DataFrame에서 요소별로 작동하며 map는 Series에서 요소별로 작동한다
"""
