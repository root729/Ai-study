import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 데이터 준비
rounds = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
scores = np.array([10, 8, 6, 4, 7, 9]) 

# 2. AI 모델 만들기
model = LinearRegression()

# 3. AI 학습시키기
model.fit(rounds, scores)

# 4. 결과 예측
next_round = np.array([[7]])
prediction = model.predict(next_round)

print(f"=== AI 분석 결과 ===")
print(f"AI 예측: 7판째에는 {prediction[0]:.2f}번 정도만에 맞출 거야!")

# 5. 시각화
plt.scatter(rounds, scores, color='black')
plt.plot(rounds, model.predict(rounds), color='red')
plt.show()
