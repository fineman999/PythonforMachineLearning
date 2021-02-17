# PythonforMachineLearning

## 비슷한 뉴스를 어떻게 선정할까?

### 컴퓨터는 문자를 그대로 이해하지 못함

문자->숫자

숫자로 유사하다는 어떻게 표현할까?

유사하다 = 가깝다.

즉 문자->숫자->벡터
  문자를 Vector로- One-hot Encoding
  ---
  - 하나의 단어를 Vector의 Index로 인식, 단어 존재시 1 없으면 0

   - Rome = [ 1, 0, 0, 0, 0, ..., 0]
  -  Paris = [ 0, 1, 0, 0, 0, ..., 0]
   - Italy = [ 0, 0, 1, 0, 0, ..., 0]
   - France = [ 0, 0, 0, 1, 0, ..., 0]
   
   Bag of words
   ---
   - 단어별로 인덱스를 부여해서, 한 문장(또는 문서)의 단어의 개수를 Vector로 표현
   
   Cosine distance
   ---
   - 두 점 사이의 각도
   - Why cosine similarity? Count > Direction
   - (Love, hate) -> (5, 0), (5 ,4), (4, 0) 어느점이 가장 가까운가?

Process
---
- 파일을 불러오기
- 파일을 읽어서 단어사전 (corpus) 만들기
- 단어별로 Index 만들기
- 만들어진 인덱스로 문서별로 Bag of words vector 생성
- 비교하고자 하는 문서 비교하기
- 얼마나 맞는지 측정하기

