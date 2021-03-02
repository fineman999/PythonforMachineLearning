import pandas as pd

"""
1. groupby_hierarchical_index
"""
# Group by - Basic
# data from:
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
         'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
         'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
         'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
         'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}

df = pd.DataFrame(ipl_data)
df
df.groupby("Team")["Points"].sum()

# Hierarchical index
df
h_index = df.groupby(["Team", "Year"])["Points"].sum()
h_index

h_index.index
h_index["Devils":"Kings"]
h_index

h_index.unstack() #unstack은 쌓은 것을 옆으로 늘어놓는것(왼쪽에서 오른쪽으로 넓게) 라고 연상이 될 것입니다.

h_index.swaplevel() #계층 순서 변경

from pandas import Series, DataFrame
import numpy as np
h_index.swaplevel().sort_index(0)
h_index.sum(level=0) #첫번쨰 인덱스 즉 team의 index

h_index.sum(level=1)# 두번쨰 인덱스, 즉 year의 index
"""
    Groupby - gropuped
"""
grouped = df.groupby("Team")
for name,group in grouped:
    print (type(name))
    print (type(group))
grouped.get_group("Riders")
df
grouped.agg(min) #여러개의 함수를 여러 열에 적용 : agg()함수
grouped.agg(np.mean) #평균값 계산
grouped['Points'].agg([np.sum, np.mean, np.std])
# agg 함수로 sum, mean, std 를 계산하여 points 값들을 보여준다.
"""
Transformation
"""
score = lambda x: (x - x.mean()) / x.std()
grouped.transform(score) #연산 결과를 채운다.

df.groupby('Team').filter(lambda x: len(x) >= 3)    #filter함수는 특정 조건으로 걸러서 걸러진 요소들로 iterator객체를 만들어서 리턴해줍니다.
df.groupby('Team').head()
df.groupby('Team').filter(lambda x: x["Points"].max() > 800)
!wget https://www.shanelynn.ie/wp-content/uploads/2015/06/phone_data.csv
df_phone = pd.read_csv("https://www.shanelynn.ie/wp-content/uploads/2015/06/phone_data.csv")
df_phone.head()

import dateutil
df_phone['date'] = df_phone['date'].apply(dateutil.parser.parse, dayfirst=True) #Pandas Dataframe(데이터프레임, 2차원) 타입의 객체에서 호출할 수 있는 apply함수
#'일(day)'이 '월(month)' 보다 먼저 나오으므로 dayfirst=True 로 설정
df_phone.head()
df_phone.groupby('month')['duration'].sum()
df_phone[df_phone['item'] == 'call'].groupby('month')['duration'].sum()
df_phone.groupby(['month', 'item'])['duration'].sum()
df_phone.groupby(['month', 'item'])['date'].head()
df_phone.groupby(['month', 'item'])['date'].count().unstack()
df_phone.groupby('month', as_index=False).agg({"duration": "sum"}) #index를 사용하고 싶은 않은 경우에는 as_index=False 를 설정하면 됩니다.
df_phone.groupby('month').agg({"duration": "sum"})


df_phone.groupby(['month', 'item']).agg({'duration':sum,      # find the sum of the durations for each group
                                     'network_type': "count", # find the number of network type entries
                                     'date': 'first'})    # get the first date per group
df_phone.groupby(['month', 'item']).agg({'duration': [min],      # find the min, max, and sum of the duration column
                                     'network_type': "count", # find the number of network type entries
                                     'date': [min, 'first', 'nunique']})    # get the min, first, and number of unique dates

grouped = df_phone.groupby('month',as_index=False).agg( {"duration" : [min, max, np.mean]})
grouped
grouped.columns = grouped.columns.droplevel(level=0)
grouped
grouped.rename(columns={"min": "min_duration", "max": "max_duration", "mean": "mean_duration"})

grouped = df_phone.groupby('month').agg( {"duration" : [min, max, np.mean]})
grouped
grouped.add_prefix("duration_") #add_prefix를 사용하면 컬럼명을 쉽게 구분할 수 있도록 prefix를 추가하여 볼 수 있습니다.

"""
2. pivot crosstab
"""

import pandas as pd
import dateutil

"""
pivot table
"""
df_phone.head()
df_phone.pivot_table(["duration"],
                     index=[df_phone.month,df_phone.item],
                     columns=df_phone.network, aggfunc="count", fill_value=0) #aggfunc: 평균값이 아닌 값들을 넣고 싶을 떄 사용, fill_value: NaN을 처리하고 싶을 때 0

"""crosstab """
df_movie = pd.read_csv("https://github.com/TEAMLAB-Lecture/AI-python-connect/blob/master/codes/ch_3/part-2/data/movie_rating.csv")
df_movie.head()
"""
Database 결측값  분석할 DataFrame을 생성했으면 결측값(missing value)이 있는지 여부에 대해서 꼭 확인하고 조치하여야 합니다.
Python pandas에서는 결측값을 'NaN' 으로 표기하며, 'None'도 결측값으로 인식합니다.
칼럼별 결측값 개수 구하기 : df.isnull().sum()
출처: https://rfriend.tistory.com/260 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]
"""


"""
3. merge_concat
"""
raw_data = {
        'subject_id': ['1', '2', '3', '4', '5', '7', '8', '9', '10', '11'],
        'test_score': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'test_score'])
df_a

raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_b
pd.merge(df_a, df_b, on='subject_id')
pd.merge(df_a, df_b, left_on='subject_id', right_on='subject_id')
pd.merge(df_a, df_b, on='subject_id', how='left')
pd.merge(df_a, df_b, on='subject_id', how='right')
pd.merge(df_a, df_b, on='subject_id', how='outer')
pd.merge(df_a, df_b, on='subject_id', how='inner')
pd.merge(df_a, df_b, right_index=True, left_index=True) #index 기준으로 합치기

"""
concatenate
"""
raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']}
df_a = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_a
raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
df_b = pd.DataFrame(raw_data, columns = ['subject_id', 'first_name', 'last_name'])
df_b
df_new = pd.concat([df_a, df_b])
df_new
df_new.reset_index()

df_a.append(df_b)
df_new = pd.concat([df_a, df_b], axis=1)
df_new = pd.concat([df_a, df_b], axis=0)
df_new.reset_index()
