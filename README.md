# 파이썬을 이용한 빅데이터분석기사 실기(2회) 준비

[체험링크](https://dataq.goorm.io/exam/116674/%EC%B2%B4%ED%97%98%ED%95%98%EA%B8%B0/quiz/1)  

## 단답형(10x3=30)
> 정답 입력 후 제출  

## 작업형
### 제1유형(3x10=30) - 데이터 처리 영역  
> 단답형 답을 가진 변수를 print 명령어로 출력하는 코드 제출
```python
# 예시문제: mtcars 데이터셋(mtcars.csv)의 qsec 컬럼을 최소최대 척도(Min-Max Scale)로 변환한 후 
# 0.5보다 큰 값을 가지는 레코드 수를 구하시오.  
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

''' 오타 방지 방법 
print(dir(sklearn))
print(dir(sklearn.preprocessing))
print(help(sklearn.preprocessing.MinMaxScaler))
'''

mtcars = pd.read_csv('data/mtcars.csv')
scale = MinMaxScaler()

print((scale.fit_transform(mtcars[['qsec']]) > 0.5).sum())
```
#### 예상 문제 

### 제2유형(1x40=40) - 모형 구축 및 평가 영역  
> write.csv, to_csv 명령어를 이용하여 답안을 csv로 생성하는 코드 제출  


```python
예상문제: 아래는 백화점 고객의 1년 간 구매 데이터이다.

(가) 제공 데이터 목록
① y_train.csv : 고객의 성별 데이터 (학습용), CSV 형식의 파일
② X_train.csv, X_test.csv : 고객의 상품구매 속성 (학습용 및 평가용), CSV 형식의 파일

나) 데이터 형식 및 내용
① y_train.csv (3,500명 데이터)
 * custid: 고객 ID
 * gender: 고객의 성별 (0: 여자, 1: 남자)

② X_train.csv (3,500명 데이터), X_test.csv (2,482명 데이터)
 고객 3,500명에 대한 학습용 데이터(y_train.csv, X_train.csv)를 이용하여 성별예측 모형을 만든 
후, 이를 평가용 데이터(X_test.csv)에 적용하여 얻은 2,482명 고객의 성별 예측값(남자일 확률)을 
다음과 같은 형식의 CSV 파일로 생성하시오.(제출한 모델의 성능은 ROC-AUC 평가지표에 따라 
채점)

<유의사항>
 성능이 우수한 예측모형을 구축하기 위해서는 적절한 데이터 전처리, Feature Engineering, 분류 
알고리즘 사용, 초매개변수 최적화, 모형 앙상블 등이 수반되어야 한다.
'''
import pandas as pd

# 1. 데이터 로드
train = pd.read_csv('data/X_train.csv')
target = pd.read_csv('data/y_train.csv')
test = pd.read_csv('data/X_test.csv')

print(train.shape, target.shape, test.shape)

df = train.merge(target, on='cust_id', how='left')
print(df.shape)

# 2. 전처리 & Feature Engineering
#print(df.info())
desc = df.describe(include="all") 

for col in df.columns:
	print(col, desc[col])
	pass
	
# null - 환불금액
#print(sum(df['환불금액'].isna())/df.shape[0])
# 65%가 null min=
#print(df['환불금액'].min())
## min 5600.0 이므로, 0으로 대체
train['환불금액'] = train['환불금액'].fillna(0)
test['환불금액'] = test['환불금액'].fillna(0)
df['환불금액'] = df['환불금액'].fillna(0)
#print(df.isna().sum().sum())

# object - 주구매상품, 주구매지점 
#print(df['주구매상품'].value_counts())
#print(df['주구매지점'].value_counts())

# 3. 분류 알고리즘 사용
# 4. 초매개변수 최적화
# 5. 모형 앙상블
# 6. 예측
# 7. 제출 

# 답안 제출 예시
# 수험번호.csv 생성
# DataFrame.to_csv("0000.csv", index=False)
```
