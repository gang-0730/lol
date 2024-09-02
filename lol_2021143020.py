import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 데이터 로드
combined_df = pd.read_csv('combined_lol_data.csv')

# 예시로 '승패'를 예측할 타겟 변수로 설정 (데이터에 따라 다를 수 있음)
target = 'win'  # 예시로 'win' 컬럼이 승패를 나타낸다고 가정

# 피처와 타겟 분리
X = combined_df.drop(columns=[target])
y = combined_df[target]

# 결측치 처리 (필요 시)
X = X.fillna(0)  # 예시로 결측치를 0으로 채움

# 범주형 변수 인코딩
X = pd.get_dummies(X)

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)