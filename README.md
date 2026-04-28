[Untitled-1.py](https://github.com/user-attachments/files/27165472/Untitled-1.py)
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 데이터 준비 (네 기록을 넣어봐. 판수가 많을수록 AI가 정확해져!)
# [1판, 2판, 3판, 4판, 5판...]
rounds = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
# [각 판별 시도 횟수]
scores = np.array([10, 8, 6, 4, 7, 9]) 

# 2. AI 모델 만들기 (선형 회귀 모델)
model = LinearRegression()

# 3. AI 학습시키기 (데이터를 머신러닝 시키는 구간!)
model.fit(rounds, scores)

# 4. 다음 판(7판째) 결과 예측하기
next_round = np.array([[7]])
prediction = model.predict(next_round)

print(f"=== AI 분석 결과 ===")
print(f"AI: '음... 네 실력을 보니 7판째에는 {prediction[0]:.2f}번 정도만에 맞출 것 같아!'")

# 5. 시각화 (AI가 그린 예측선 확인)
plt.scatter(rounds, scores, color='black', label='Actual Data') # 실제 데이터
plt.plot(rounds, model.predict(rounds), color='red', linewidth=2, label='AI Prediction Line') # AI가 찾은 선
plt.title("AI Performance Prediction")
plt.xlabel("Game Round")
plt.ylabel("Attempts")
plt.legend()
plt.show()
