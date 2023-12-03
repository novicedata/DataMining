📌 LDA, QDA
LDA와 QDA는 다변량 정규 분포를 위한 베이즈 분류기이다.(베이즈 이론에 대해 알고 보시는게 이해가 쉽습니다.)



다변량 정규 분포는 두 개 이상의 확률변수 가령 $X, Y$가 상호의존적으로 정규 분포를 따를 때의 확률 분포입니다. 다변량 정규 분포의 확률 밀도 함수를 보면 원래 정규분포와 비슷한 형태를 보입니다.



그럼 이둘은 무슨 차이가 있을 까요? 큰 차이는 결정 경계. 즉, 분류를 하기 위한 경계를 어떻게 모델링하느냐에 있습니다.



LDA, QDA의 이름부터 Linear, Quadratic인 것을 보시면 아마 금방 유추가 가능할겁니다.



QDA의 경우 각 클래스마다 고유한 공분산 행렬을 사용합니다.

예를 들어 클래스가 3개 있다면 $\Sigma_1 \neq \Sigma_2 \neq \Sigma_3$인 것이죠.

또한 각 클래스의 결정 경계가 선형이 아닌 이차함수와 같은 비선형 결정 경계로 나타납니다.



반면 LDA는 각 클래스에서 공분산 행렬이 동일하다고 가정하고 선형 결졍 경계를 가집니다.



아래 그림은 파이썬으로 그린 예시 입니다.



그래프로 보시면 왼쪽 QDA의 경우 좀 더 데이터에 적합한 그래프를 그리는 것을 보실 수 있습니다. 반면 오른쪽은 LDA의 경우 선형 경계로 그려진 것을 보실 수 있습니다.








◼️ QDA(Quadratic Discriminant Analysis)


각각의 클래스 $Y=i$에 대해, 예측 변수 벡터 $\textbf{X} = (X_1,...,X_p)^T$의 다변량 정규 분포를 가정하고, 이 분포는 평균 벡터 $\pmb{\mu}$, 공변량 매트릭스 $\pmb{\Sigma}_i$를 가집니다.



클래스 $Y=i$에서 예측 변수 벡터의 분포는 다변량 정규 분포 밀도 함수를 따르고 아래와 같습니다.

$$
f(\textbf{x}|Y=i) = f_i(\textbf{x}) = \dfrac{1}{(2\pi)^{p/2}|\pmb{\Sigma}_i|^{1/2}}
\exp \left(-\dfrac{1}{2}(\textbf{x}-\pmb{\mu}_i)^T \pmb{\Sigma}_i^{-1} (\textbf{x}-\pmb{\mu}_i) \right)
$$



이에 로그를 취해주면

$$
\log f_i(\textbf{x}) = -\dfrac{1}{2}(\textbf{x}-\pmb{\mu}_i)^T \pmb{\Sigma}_i^{-1} (\textbf{x}-\pmb{\mu}_i) \dfrac{1}{2} \log |\pmb{\Sigma}_i| - \dfrac{p}{2} \log (2\pi)
$$



그럼 베이즈 분류기는 $\textbf{x}$를 어떤 클래스에 할당하냐면 아래를 최대화하는 클래스에 할당합니다. ($p_i$는 사전확률)



$$
\hat{Y}(\textbf{x}) = \arg\max_{1\leq i \leq K}(p_if_i(\textbf{x})) = \arg\max_{1\leq i \leq K}(\log p_i+\log f_i(\textbf{x}))
$$



위 식을 통해 분류 방정식 $\delta_i(\textbf{x})$를 정의하면 아래와 같습니다.(그냥 위식을 대입한것)

$$
\delta_i(\textbf{x}) = -\dfrac{1}{2}(\textbf{x}-\pmb{\mu}_i)^T \pmb{\Sigma}_i^{-1} (\textbf{x}-\pmb{\mu}_i) \dfrac{1}{2} \log |\pmb{\Sigma}_i| - \log p_i
$$



이와 같이 베이즈 분류기는 $\hat{Y}(\textbf{x}) = \arg\max_{1\leq i \leq K} \delta(\textbf{x})$에 $\textbf{x}$의 클래스를 할당합니다.



통계를 아시는 분이라면 위 식에서 $\pmb{\mu}, \pmb{\Sigma}_i, p_i$를 알 수 없다는 것을 아실겁니다. 주어진 데이터를 이용하여 평균, 공분산 행렬을 추정하여 추정식을 이용해야 합니다.



$$
\hat{\pmb{\mu}}_i = \bar{\textbf{x}}_i, \ \ \hat{p}_i = \frac{n_i}{n}
$$





◼️ LDA(Linear Discriminant Analysis)


위에서 설명한 것과 같이 LDA의 경우 공분산 매트릭스를 모두 같다고 하기 때문에, 예측 변수 벡터는 평균 벡터$\pmb{\mu}_i$와 공분산 매트릭스 $\pmb{\Sigma}(\neq \pmb{\Sigma}_i)$를 가진 정규 분포를 가정합니다.



위 QDA와 같이 전개하나 상수를 제거한다면 아래와 같은 분류 방정식이 나옵니다.



$$
\delta_i(\textbf{x}) = \pmb{\mu}_i\pmb{\Sigma}^{-1}\textbf{x}-\dfrac12 \pmb{\mu}_i\pmb{\Sigma}^{-1}\pmb{\mu}_i +\log p_i
$$



QDA처럼 추정식을 이용하며 공분산 행렬의 추정을 보면



$$
\hat{\pmb{\Sigma}} = \textbf{S}_p = \dfrac{\Sigma_{i=1}^K (n_i-1) \textbf{S}_i}{\Sigma{i=1}^K (n_i-1)}
$$



이와 같이 베이즈 분류기는 $\hat{Y}(\textbf{x}) = \arg\max_{1\leq i \leq K} \hat{\delta}(\textbf{x})$에 클래스를 할당합니다.
