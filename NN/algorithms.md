## 📌 경사 하강법 최적화 알고리즘

경사 하강법 최적화 알고리즘의 변형들로, 신경망의 학습 속도와 성능을 향상 시키기 위해 고안되었다.

### ◼️ Momentum method

**모멘텀을 사용하는 확률적 경사 하강법은 각 반복에서의 업데이트인 $\\Delta w$ 을 기억하고, 다음 업데이트를 그래디언트와 이전 업데이트의 선형 조합으로 결정합니다.**

**이전 단계에서의 속도(momentum)을 사용해 다음 단계의 속도를 결정하면서 최적화를 수행하는 것입니다.**

$$  
\\begin{align}  
\\Delta \\textbf{w}^{r+1} &= \\alpha \\Delta \\textbf{w}^r - \\eta \\nabla E\_i ( \\textbf{w}^r)  
\\\\ \\textbf{w}^{r+1} &= \\textbf{w}^r + \\Delta\\textbf{w}^{r+1}  
\\\\ &= \\textbf{w}^r - \\eta \\nabla E\_i(\\textbf{w}^r) + \\alpha \\Delta \\textbf{w}^r  
\\end{align}  
$$

$\\alpha$값은 0~1 사이의 지수적 감소 계수로, 현재 그래디언트와 이전 그래디언트가 가중치 변경에 얼마나 기여하는지를 결정합니다.

이러한 접근은 경사 하강법에서의 진동과 수렴의 문제를 완화하고 빠르게 수렴할 수 있도록 도와줍니다.

### ◼️ AdaGrab(Adaptive Gradient)

AdaGrab은 각 time step $r$에서 모든 파라미터 $w\_i$에 대해 다른 학습률을 사용합니다.

**많이 변하지 않는 파라미터는 큰 학습률을, 자주 변하는 파라미터는 작은 학습률을 갖게 됩니다. 각각의 매개변수에 맞춰서 학습률을 변화시키고, 학습률을 점차 줄여가는 방법이다.**

아래 식에서 $G^r = \\sum\_{t=1}^r \\text{diag} ((\\textbf{g}^t)(\\textbf{g}^t)^T) \\in R^{d\\times d}$이고 이는 각 대각 성분 $(i,i)$이 시간 단계 $r$까지의 $w\_i$에 대한 그래디언트의 제곱 합인 대각 행렬이다.

$\\textbf{g}^t$는 시간 단계 $t$에서의 그레디언트 벡터.(각 파라미터에 대한 편미분 값을 담고 있는 벡터)

$$  
\\textbf{g}^t = \\begin{bmatrix} \\frac{\\partial E}{\\partial w\_1} \\\\ \\frac{\\partial E}{\\partial w\_2} \\\\ \\vdots \\\\ \\frac{\\partial E}{\\partial w\_d} \\end{bmatrix}  
$$

$E$는 손실함수이며, 각 $w\_i$는 파라미터 가중치 이들의 편미분 벡터

$$  
G^r = \\sum\_{t=1}^r \\text{diag}((\\textbf{g}^t)(\\textbf{g}^t)^T) =  
\\begin{bmatrix}  
\\sum\_{t=1}^r (\\textbf{g}^t\_1)^2 & 0 & \\cdots & 0 \\\\  
0 & \\sum\_{t=1}^r (\\textbf{g}^t\_2)^2 & \\cdots & 0 \\\\  
\\vdots & \\vdots & \\ddots & \\vdots \\\\  
0 & 0 & \\cdots & \\sum\_{t=1}^r (\\textbf{g}^t\_d)^2  
\\end{bmatrix}  
$$

$\\epsilon$은 smooting term으로 0으로 나눠지는 걸 방지한다.(일반적으로 $10^{-8}$)정도를 사용.

($\\odot$ : 행렬 요소별 곱셈)

$$  
\\textbf{w}^{r+1} = \\textbf{w}^r = -\\dfrac{\\eta}{\\sqrt{G^r + \\epsilon}} \\odot \\textbf{g}^r  
$$

**문제는 제곱 그래디언트의 합($G^r$)이 시간에 따라 계속해서 누적되기 때문에 학습률이 너무 작아지게 되고, 결국 가중치의 업데이트가 거의 이루어지지 않는 단계가 생김.**

### ◼️ RMSProp(Root mean square propagation)

RMSProp는 Adagrab의 극심한 학습률 감소를 해결하기 위해 개발되었다.

학습률을 각 매개변수에 대해 적응적으로 조절하는 방식이다. 각 매개변수마다 다른 학습률을 사용하여 효율적으로 학습한다.

**Adagrab과 다르게 지수 가중 이동 평균(Exponential moving Average)을 사용하여 과거의 제곱된 그래디언트 값을 추적한다. 최근 그레디언트 정보에 더 높은 가중치를 부여하면서 그래디언트 변화를 추적하는데 도움이 된다.**

$$  
\\begin{align}  
E\[g^2\]^r &= \\gamma E\[g^2\]^{r-1} + (1-\\gamma)\[g^2\]^r  
\\\\ &= \\textbf{w}^{r+1} = \\textbf{w}^r - \\dfrac{\\eta}{\\sqrt{E\[g^r\]^r +\\epsilon}} \\textbf{g}^r  
\\end{align}  
$$

이렇게 각 가중치에 대한 제곱 그래디언트의 지수적으로 감소하는 평균으로 학습률을 나는다.

### ◼️ Adam(Adaptive momentum)

적응적 모멘텀 추정은 각 매개변수에 대한 적응적 학습률을 계산하는 또 다른 방법이다. RMSProp과 같이 지수적으로 감소하는 **과거 제곱 그래디언트의 평균 $v^r$**을 저장하는 것 외에도, Adam은 모멘텀과 유사하게 **과거 그래디언트의 지수적으로 감소하는 평균 $\\textbf{m}^r$**도 유지한다.

RMSProp과 비슷하게 각 매개변수에 대한 제곱 그래디언트의 지수적으로 감소하는 평균을 계산하여 가중치 업데이트에 사용하고,

모멘텀과 비슷하게, 과거의 그래디언트의 지수적으로 감소하는 평균 $\\textbf{m}^r$도 유지한다는 것.

$$  
\\begin{align}  
\\textbf{m}^r &= \\beta\_1 \\textbf{m}^{r-1} + (1-\\beta\_1)\\textbf{g}^r  
\\\\ v^r &= \\beta\_2 v^{r-1} + (1-\\beta\_2)(g^2)^r  
\\end{align}  
$$

**$\\textbf{m}^r, v^r$이 0으로 초기화된 벡터로 시작하기 때문에 초기 단계에서(감쇠율이 작을 때) 이 들이 편향 되어있다. 그래서 편향을 상쇄하기 위해 편향 보정된 첫, 두 번째 모멘트 추정값을 계산하여 사용한다.**

$$  
\\begin{align}  
\\hat{\\textbf{m}}^r &= \\dfrac{\\textbf{m}^r}{1-\\beta\_1^r}  
\\\\ \\hat{v}^r &= \\dfrac{v^r}{1-\\beta\_2^r}  
\\end{align}  
$$

다음 이러한 값을 사용해서 매개변수를 업데이트하며, Adam update rule을 생성

$$  
\\textbf{w}^{r+1} = \\textbf{w}^r - \\dfrac{\\eta}{\\sqrt{\\hat{v}^r + \\epsilon}} \\hat{\\textbf{m}^r}  
$$
