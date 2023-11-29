# ECM(Expected cost of misclassification)과 베이즈 분류

ECM을 설명하기 전에 몇가지 가정과 명명을 확인하겠습니다,

$f_i$를 밀도 함수라고 하고 클래스 $i$가 1과 2가 있다고 하자. 이 때 $f_i(x) = f(x|Y=i)$는 클래스 $i$가 주어졌을 때의 조건부 밀도 일 것이다.

$P(j|i)$는 실제 클래스 $i$를 클래스 $j$로 오분류할 확률이라고 하자.

이 때 $c(1|2), c(2|1)$을 오분류 비용(misclassification cost)라고 정의하자. [$c(1|2)$ 라면 클래스 2에서 클래스 1로의 오분류에 따라 얻는 우리의 손실값이다.]

또한 $p_1, p_2$는 각 클래스의 사전 확률이다.

Expected cost of misclassification(ECM)은 아래와 같다.

$$
ECM = c(2|1)P(2|1)p_1 + c(1|2)P(1|2)p_2
$$

$R_1, R_2$를 각각 클래스 1과 2의 분류 영역이라고 정의할 때, 우리는 ECM을 최소화하는 이 분류 영역을 찾을 수 있다.

$$
\begin{align}
ECM &= c(2|1)P(2|1)p_1 + c(1|2)P(1|2)p_2
\\ &= c(2|1)p_1 \int_{R_2} f_1(\textbf{x})d\textbf{x} + c(1|2)p_2 \int_{R_1} f_2(\textbf{x})d\textbf{x}
\\ &= c(2|1)p_1 + \int_{R_1} \left(c(1|2)p_2f_2(\textbf{x}) - c(2|1)p_1f_1(\textbf{x})\right) d\textbf{x}
\end{align}
$$

이를 통해 $R_1, R_2$를 아래와 같이 정의 가능하다.

$$
\begin{align}
R_1 &= \{\textbf{x} : c(1|2)p_2f_2(\textbf{x}) - c(2|1)p_1f_1(\textbf{x}) <0\}
\\ &= \dfrac{f_1(\textbf{x})}{f_2(\textbf{x})} > \dfrac{c(1|2)p_2}{c(2|1)p_1}
\\ R_2 &= \{\textbf{x} : c(1|2)p_2f_2(\textbf{x}) - c(2|1)p_1f_1(\textbf{x}) >0\}
\\ &= \dfrac{f_1(\textbf{x})}{f_2(\textbf{x})} < \dfrac{c(1|2)p_2}{c(2|1)p_1}
\end{align}
$$

Total probablity of misclassification(TPM)

분류를 좀더 정확성있게 하려면 당연히 오분류의 전체 확률을 최소화 해야한다. TPM이 아래와 같을 때

$$
TPM = p_1 \int_{R_2} f_1(\textbf{x}) d\textbf{x} + p_2 \int_{R_1} f_2(\textbf{x}) d\textbf{x}
$$

각 $R_1, R_2$는

$$
R_1 : \dfrac{f_1(\textbf{x})}{f_2(\textbf{x})} > \dfrac{p_2}{p_1} \leftrightarrow  p_1f_1(\textbf{x}) > p_2f_2(\textbf{x})
\\ R_2 : \dfrac{f_1(\textbf{x})}{f_2(\textbf{x})} < \dfrac{p_2}{p_1} \leftrightarrow  p_1f_1(\textbf{x}) < p_2f_2(\textbf{x})
$$

이다. (분자인 $f(\textbf{x})$ 생략... $\dfrac{p_1 f_1(\textbf{x})}{f(\textbf{x})} \propto p_1f_1(\textbf{x})$)

앞서 포스팅한 내용과 크게 다른 내용은 없다. 똑같이 베이즈 분류기는 전체 오분류 확률을 최소화하는 방법을 사용한다는 것.

이를 최대 사후 확률 분류. Maximum posterior probability(MPP) classification 이라고도 한다.

# 베이즈 오류율(Bates error rate)

베이즈 분류기는 불가능한 테스트 오류율인 Bayes error rate를 생성하는데, 기본적인 error와 유사한 개념이다.

각 $\textbf{X}$에서 분류 오류는 $1-\max_j Pr(Y = j|\textbf{X})$로 정의된다. 데이터에 대해 가장 확률이 높은 클래스를 선택하는 방법에서 발생되는 최소 오차이다.

$$
\begin{align}
\text{Bayes' error rate} &= 1-E\left(\max_j Pr(Y=j|\textbf{X})\right)
\\ &= 1- E\left(\max_j \tau_j(\textbf{X}) \right)
\end{align}
$$

## 예를 들어보자.

$Y$는 이분 반응 변수이며 {1,2}로 구성되어있다. $Y=1$의 사전확률인 $p_1 =0.3$이고, $Y=1$ 조건일 때 $X$는 평균이 1, 분산이 1인 정규 분포라고 가정하자. 자연스럽게 $Y=2$의 사전 확률은 $p_2 =0.7$

$Y=2$일 때 $X$는 평균이 3이고 분산이 1인 정규 분포다. 이 때 Bayes error rate를 구해보자. 우선 앞서 확인한 것 처럼 베이즈 분류기는 다음과 같은 조건으로 관측값을 $Y=2$로 분류하한다.

$$
p_1f_1(x) \leq p_2f_2(x)
$$

우리는 각각 $f_1(x), f_2(x)$의 밀도함수를 알고 있다. 또한 각각의 사전확률을 이용하여 풀어주고, 이를 만족하는 임계값 $x$를 찾는다면 아래와 같다.

$$
\begin{align}
0.3 \times \dfrac{1}{\sqrt{2\pi}}\exp\left[-\dfrac{(x-1)^2}{2}\right]
&\leq 0.7 \times \dfrac{1}{\sqrt{2\pi}}\exp\left[-\dfrac{(x-3)^2}{2}\right]
\\ x &\geq 2-\dfrac12 \log \left(\dfrac{0.7}{0.3}\right) = 1.576351
\end{align}
$$

밀도 함수가 정규 분포이기 때문에 이의 누적 분포함수인 표준 정규분포를 이용하여 계산하면 베이즈 에러율은 다음과 같다.

$$
p_1P(Z> \dfrac{1.576351-1}{1}) + p_2P(Z> \dfrac{1.576351-3}{1}) = 0.1387485
$$

# 사후 확률 $\tau_j(\textbf{x})$를 구하는 법?

이에는 여러가지가 있는데 우선,

1. 선형 판별 분석(LDA) 및 이차 판별 분석(QDA)에서는 구성 밀도(component density)가 다변량 정규 분포를 따른다고 가정한다. 
  다변량 정규 분포는 여러 변수에 대한 확률 분포를 나타낸다.
  

2. 한편 로지스틱 회귀 모델의 경우 사후 확률의 형태를 parametric form이라고 가정한다. 이는 데이터의 feature와 class 간의 관계를 설명하는 매개변수화된 모델을 의미한다. 아마 로지스틱 회귀를 아는 사람이라면 금방 알 것입니다. 아래와 같아요.
  

$$
\tau_1(\textbf{x}) = P(Y=1|\textbf{x}) = \dfrac{\exp(\beta_0 + \pmb{\beta'}\textbf{x})}{1 + \exp(\beta_0 + \pmb{\beta'}\textbf{x})}
$$

3. 나이브 베이즈 분류기(Naive Bayes)는 조건부 독립을 가정하고 이를 통해 추정을 쉽게 수행한다. 즉, feature들 사이의 상관이 없고 모든 변수가 독립적으로 기여한다고 가정하여 이 변수들 간의 결합 확률을 단순히 단일 변수들의 조건부 확률의 곱으로 타나낼 수 있다.
  
4. 최근접 이웃(KNN)의 경우 $\textbf{x}_0$ 근처의 사후 확률을 근사화 하여 사용한다. 각 최근접 이웃의 클래스를 합하여 이웃의 수로 나누어 클래스 $j$의 사후확률을 계산하는 법이다.
  

$$
\tau_j(\textbf{x}_0) = \dfrac1K \sum_{\textbf{x}_m \in N_K(\textbf{x}_0)} I(y_m =j)
$$
