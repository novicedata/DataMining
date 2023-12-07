# Stacking

앙상블 모델 중 하나인 stacking 입니다. stacking이 재밌는 점은 여러 모델들을 학습하고 이를 또다시 새로운 모델로 학습한다는 점입니다.

기본 모델들이 각자의 예측을 수행 한뒤, 그 예측값들을 이용해 새로운 모델을 학습합니다. 이를 통해 기본 모델이 갖는 각자의 강점을 살려 높은 성능을 보일 수 있습니다.

스태킹에서는 크게 세 가지 구성 요소가 있는데

기본 모델

기본적인 모델로 서로 다른 알고리즘을 사용하거나, 다양한 하이퍼 파라미터로 설정된 모델들(로지스틱 회귀, 서포트 벡터 머신, 랜덤 포레스트 등등)

First Level 예측

각 기본 모델이 주어진 데이터에 대해 각자 예측을 수행합니다. 이 예측한 값들이 First Level에서 쌓(stack!)입니다.

Second Level 모델

이렇게 쌓인 예측값들을 이용하여 Second Level 에서 새로운 모델을 학습합니다. 이 모델은 각 예측값의 중요도를 학습하여 최종 예측을 수행합니다.

정말 신기한 발상이 아닌가요? 모델들의 예측값을 다시 데이터로 이용하여 새로운 모델로 학습한다.. 인간의 상상력을 뛰어난 것 같아요. 하지만 아쉽게도 항상 좋은 결과가 나오는 것은 아닙니다.(애초에 FIrst Level에서 예측이 잘 안됐다면.. 이 예측값을 사용하는게 의미가 있기 어렵겠죠?)

실제로도 stacking을 사용하는 경우는 많지 않다고 하지만, 대회에서는 조금의 성능차이도 유의미 하기 때문에 종종 사용된다고 합니다.

아래 그림으로 전체적인 개괄을 본다면.. 이해가 쉬울 것입니다.

![](file://C:\Users\atlsw\AppData\Roaming\marktext\images\2023-12-07-19-45-34-image.png?msec=1701945934929)

수학으로 보면??

$f_1,...,f_m$은 $m$ 개의 모델입니다. 우리는 $k$개의 모델을 합쳐 최적의 $f(\textbf{x}) = \sum^m_{j=1} \alpha_j, f_j(\textbf{x})$를 찾고 싶습니다. 가장 쉬운 방법은 역시 잔차를 최소화하는 것이겠죠? 아래처럼 RSS를 구하는 것이 가장 쉬울 겁니다(Residual sum of squares)

First level 예측 :

$$
\sum_{i=1}^N \left(y_i - \sum^m_{j=1}\alpha_j f_j(\textbf{x})  \right)^2
$$

중요한 점은 최적의 $f(\textbf{x})$를 찾기 위해서 당연히 동일한 train set를 사용하면 안됩니다! 과적합의 지름길입니다. 각각의 $\alpha_j, f_j(\textbf{x})$모두 validation set를 이용합시다.

Data stack :

그러면 $z_{ij} = \hat{f}_j^{-k(i)}(\textbf{x}_i)$를 $i$번째 데이터인 $(y_i, \textbf{x}_i)$를 제외한 데이터로 훈련된 $\textbf{x}$에서의 예측값이라고 합시다. 식만 어렵지 train set에서 얻은 $j$번째 모델의 에측값이라는 겁니다.

이를 통해 First level 예측으로 얻은 데이터인 $(y_i, \textbf{z}_i)$를 얻을 수 있습니다.

Second Level model :

이 데이터를 이용하여 새로운 모델로 또다시 예측합니다.

$$
\sum_{i=1}^N \left(y_i - \sum_{j=1}^m \alpha_j z_{ik}\right)^2
$$

아래 코드는 간단한 stacking 예제 입니다. 두개의 모델만을 사용했고(선형회귀, 결정트리) 마지막 모델도 선형회귀를 사용하였습니다.

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 생성 (예시로 사용할 데이터)
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Level One 모델로 Linear Regression과 Decision Tree Regressor를 사용
model1 = LinearRegression()
model2 = DecisionTreeRegressor()

# Training set을 이용하여 Level One 모델들 학습
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Validation set을 이용하여 Level One 모델들의 예측값 구하기
pred1 = model1.predict(X_val)
pred2 = model2.predict(X_val)

# Level Two 모델로 Linear Regression 사용
final_model = LinearRegression()

# Level One 모델들의 예측값과 Validation set을 이용하여 Level Two 모델 학습
X_stack_final = np.column_stack((pred1, pred2))
final_model.fit(X_stack_final, y_val)

# 최종 예측값 구하기
final_pred = final_model.predict(X_stack_final)

# 계산 및 출력
mse_final = mean_squared_error(y_val, final_pred)
rmse_final = np.sqrt(mse_final)
r2_final = r2_score(y_val, final_pred)

# 결과 출력
print(f"Level One 모델 1 예측값: {pred1}")
print(f"Level One 모델 2 예측값: {pred2}")
print(f"Level Two 모델(최종 모델) 예측값: {final_pred}")
print("\n평가 지표:")
print(f"MSE (최종 모델): {mse_final}")
print(f"RMSE (최종 모델): {rmse_final}")
print(f"R^2 (최종 모델): {r2_final}")
```

Level One 모델 1 예측값: [ -58.20320564 -0.99412493 84.42053424 -19.80868937 67.78544042
 115.08677163 195.3800306 -126.83541741 -185.65676039 -55.87134508
 80.26682995 -139.17457332 116.77784102 -42.44949517 112.75540523
 77.47491592 -27.1998263 2.03152121 -74.29575855 43.73414793]

Level One 모델 2 예측값: [ -58.16543069 -8.50598757 38.00198674 -18.26020326 36.05636659
 159.51139025 198.80943896 -120.4491733 -184.17305676 -14.15178396
 68.08398551 -120.4491733 145.65944483 -39.57740391 108.79318168
 104.63431719 -14.15178396 -8.66011001 -85.69188608 47.5813426 ]

Level Two 모델(최종 모델) 예측값: [ -58.20860893 -1.01516327 84.30008778 -19.80648137 67.70393039
 115.21912755 195.40676917 -126.82951919 -185.66964653 -55.76181019
 80.24024314 -139.13582151 116.86756712 -42.44568021 112.75436747
 77.55636692 -27.16663069 2.00200272 -74.33408077 43.74839792]

평가 지표:
MSE (최종 모델): 0.011832079478251361
RMSE (최종 모델): 0.1087753624597563
R^2 (최종 모델): 0.9999987372273896
