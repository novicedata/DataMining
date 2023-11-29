# 통계적 의사 결정 for Classification

분류에 대해서는 일반적으로 zero-one loss function을 사용하는 것이 흔한 method다.

$$
L(a,b) = I(a \neq b)
$$

$Y=1,...,K$이고 $K$는 가능한 범주들 이라고 하자. 즉, $Y$는 $K$개의 가능한 범주 중 하나.

이 때 실제 값 $Y$와 예측 값 $f(\bold{X})$의 기댓값은 아래와 같다. 주어진 입력 $\bold{X}$에 대한 y의 조건부 확률을 곱해준다.

$$
E[L(Y, f(\bold{X}))] = E_{\bold{X}} \left[\sum_{y=1}^K L(y, f(\bold{X})) p(y|\bold{X}) \right]
$$

실제 클래스 $Y=i$라 하고 예측 클래스 $f(\bold{X}) =j$라고 할 때. point 별로 예상되는 예측 오류를 최소화 할 때 아래식이 도출된다.

$$
\begin{align}
\min_{j \in 1,...,K} \sum_{i=1}^K L(i,j)p(i|X) &= \min_{j \in 1,...,K} \left[ \sum_{i=1}^K I(i \neq j) p(i|\bold{X}) \right]
\\ &= \min_{j \in 1,...,K} \left[ \sum_{i=1}^K (1-I(i=j)) p(i|\bold{X})\right]
\\ &= \min_{j \in 1,...,K} \left[ \sum_{i=1}^K p(i|\bold{X}) - \sum_{i=1}^KI(i=j) p(i|\bold{X}) \right]
\\ &= \min_{j \in 1,...,K}[1-p(j|\bold{X})]
\end{align}
$$

3번 째 식에서 결국 모든 클래스에 대한 조건부 확률의 총 합에서 옳은 예측을 하는 클래스의 조건부 확률만을 총 합한 값을 빼준다.

이는 결국 1에서 각 클래스 $j$에 대한 조건부 확률 $p(j|\bold{X})$를 빼는 것과 같다.

이 식을 최소화하는 $j$를 찾는 것이기 때문에 입력 $\bold{X}$에 대한 클래스 $j$의 확률이 최대한 1에 가까워야한다.

결국 이는 아래 식과 같다

$$
\begin{align}
f(\bold{X}) &= \arg \min_{y \in 1,...K} [1-p(y|\bold{X}=\bold{x})]
\\ &= \arg \max_{y \in 1,..., K} p(y|\bold{X}=\bold{x})
\end{align}
$$

이를 우리는 베이즈 분류기라고 부른다.

베이즈 정리를 사용하여 입력 데이터에 대한 각 클래스의 사후 확률을 계산하고, 그 중에서 가장 높은 확률을 가진 클래스를 선택하는 분류 방법이다. 위 예시처럼 zero-one loss function을 최소화하는 접근이 베이즈 분류기에서 많이 사용되는 접근 방식이다.

# 오분류에 대한 손실 행렬, 비용 행렬

오분류에 할당된 가중치를 조절하기 위한 방법.

$L_{ij} = c(j|i)$를 실제 클래스 $i$를 예측 클래스 $j$로 오분류 했을 때의 비용이라고 하자. $\bold{X} = \bold{x}$가 주어졌을 때, 관측치 $y=j$ 클래스로 분류하거나 예측할 때 손실 혹은 비용

은 아래와 같다.

$$
\sum_{i=1}^K c(j|i)p(i|\bold{x})
$$

우리는 이 손실 혹은 비용을 당연히 최소화하는 예측 클래스 $j$를 골라야한다. 이를 식으로 표현하면

$$
\begin{align}
\hat{Y}(\bold{x}) &= \arg \min_{j \in 1,...,K} \sum_{i=1}^K c(j|i)p(i|\bold{x})
\end{align}
$$

## 예를 즐어보자.

우리에게 세가지 클래스가 있고 각각의 사전 확률이 $p_1 = 0.3, p_2 = 0.6, p_3 = 0.1$ 이다. 각각 $Y$가 $1,2,3$ 일 때, $\bold{x}_0$에서의 조건부 밀도는 $f_1(\bold{x}_0) = 2, f_2(\bold{x}_0) = 0.7, f_3(\bold{x}_0) = 0.1$ $Y=1$로 주어졌을 때 $\bold{x}_0$에서의 사후 확률은 아래와 같을 것이다.

$$
\dfrac{p_1 f_1(\bold{x}_0)}{\sum_{i=1}^3 p_i f_i(\bold{x}_0)} 
= \dfrac{0.3 \times 2}{0.3 \times 2 + 0.6 \times 0.7 + 0.1 \times 0.1}
 = \dfrac{60}{103}
$$

이를 통해 각각의 사후 확률을 구하면 아래와 같다.

$$
\tau_1(\bold{x}_0) = \dfrac{60}{103}, \ \ \tau_2(\bold{x}_0) = \dfrac{42}{103}, \ \ \tau_3(\bold{x}_0) = \dfrac{1}{103}
$$

베이즈 분류기가 관측치 $\bold{x}_0$를 몇 번째 클래스로 분류하는지 알아보기는 쉽다. 앞서 베이즈 분류기는 아래 식 처럼 사후 확률이 가장높은 클래스에 할당하기 때문에 $\hat{Y}(\bold{x}_0) =1$로 분류하게 된다.

$$
\begin{align}
f(\bold{x_0}) &= \arg \max_{y \in 1,2,3} p(y|\bold{x}_0)
\end{align}
$$

그렇다면 손실 행렬을 통해 분류하면 결과가 달라질까? 우선 분류에 대한 cost를 각각 아래와 같이 얻었다고 가정하자.

실제 클래스 $i$, 예측 클래스 $j$일 때 각 코스트 $c(j|i)$

$$
c(1|1) = 0, \ \ c(1|2) = 10, \ \ c(1|3) = 1
\\ c(2|1) = 5, \ \ c(2|2) = 0, \ \ c(2|3) = 10
\\ c(3|1) = 1, \ \ c(3|2) = 10, \ \ c(3|3) = 0

$$

오분류에 대한 손실 행렬을 통해 예측값을 구하는 식은 아래와 같았다.

$$
\begin{align}
\hat{Y}(\bold{x}) &= \arg \min_{j \in 1,...,K} \sum_{i=1}^K c(j|i)p(i|\bold{x})
\end{align}
$$

이를 통해 각각의 예측값에 대해 $\hat{Y}$를 구하면 아래와 같다. 위에서 부터 각각 1,2,3으로 예측했을 때이다.

$$
c(1|2)\tau_2(\bold{x}_0) + c(1|3)\tau_3(\bold{x}_0) = \dfrac{10\times 42 + 1\times 1}{103} = \dfrac{421}{103}
\\
\\ c(2|1)\tau_1(\bold{x}_0) + c(2|3)\tau_3(\bold{x}_0) = \dfrac{5 \times 60 + 10 \times 1}{103} = \dfrac{310}{103}
\\
\\ c(3|1)\tau_1(\bold{x}_0) + c(3|2)\tau_2(\bold{x}_0) = \dfrac{1 \times 60 + 10 \times 42}{103} = \dfrac{480}{103}
$$

결과적으로 class 2번으로 분류했을 때 cost가 가장 작다. 때문에 위 방식으로 구분했을 때는 $\hat{Y}(\bold{x}_0) =2$로 베이즈 분류기와 다른 예측값이 나온다.
