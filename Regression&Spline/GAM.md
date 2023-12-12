## 📌 GAM(Generalized additive Model)

---

통계 모델링에서 사용되는 강력한 방법 중 하나로 다양한 예측 변수들을 효과적으로 다루면서 비선형성과 상호 작용을 모형화(여기선 안다룸)할 수 있는 모델.

**다변량 함수 형태로 각 변수의 영향을 표현하고, 이러한 함수들을 결합하여 종속 변수와의 관계를 모델링한다.**

일반화 선형 모델(Generalized Linear Model, GLM)의 확장 모형으로, **여러 예측 변수들 간의 비선형 관계를 고려**할 수 있다.

기존 선형 확장 모델에서 $\\beta\_0 + \\Sigma\_j \\beta\_j X\_j$의 모델을 사용했다면 함수를 결합하기 때문에 아래와 같은 모델로 사용한다.

$$  
\\beta\_0 + \\Sigma\_j f\_j (X\_j)  
$$

**각각의 $f\_j$(for each $X\_j$)에 대해 계산한 후, 이들을 결합하여 사용하는 것.**

앞서 포스팅 했던 Natural Spline, smoothing Spline, local regression, polynomial regression 등이 $f\_j$로 사용될 수 있다.

**이게 중요한 점이 뭐냐면 위 함수들은 비선형 관계를 모델링하기 위해 사용되었었다는 점.** 때문에 $f\_j$는 비선형 함수로 $\\beta\_j$를 대체하는 것이다.

예를 들면? 변수 $x\_1, x\_2, x\_3$이 각각 age, gender, height이라고 하자 $Y$를 호감도라고 하겠다 그렇다면 아래와 같이 가법 모형을 사용할 수 있는 것이다.(변수들의 비선형을 가정하자)

$$  
Y = \\beta\_0 + f\_1(age) + f\_2(gender) + f\_3(height) + \\epsilon  
$$

이 때의 $f\_1, f\_2, f\_3$은 각각의 비선형 모델을 뜻한다.

### ◾ In Regression

회귀에서 이를 계산하기 위해 **Backfitting Algorithm**을 사용하는데, 각 변수에 대한 추정치를 반복적으로 업데이트 함으로써 모델 파라미터를 조정하는 iterative optimization algorithm이다. 다른 변수들은 고정하고, 기저함수(Basis function)을 기반으로 모델링된 각 feature들을 반복적으로 갱신하는 방식이다.

아래식에서 $S\_j$는 smoothing이다.

$(1) \\text{ Initialize } \\hat{\\beta}\_0 = \\dfrac1N \\sum\_{i=1}^N y\_i, \\hat{f}\_j \\equiv 0 \\ \\forall i,j  
\\\\ (2) \\text{ Cycle }: j =1,2,...,p,...,1,2,...,p,...,$

$$  
\\begin{align}  
\\hat{f}\_j &\\leftarrow S\_j \\left\[(y\_i - \\hat{\\beta}\_0 - \\sum\_{k\\neq j}\\hat{f}\_k(x\_{ik}))^N\_1 \\right\]  
\\\\ \\hat{f}\_j &\\leftarrow \\hat{f}\_j - \\dfrac1N \\sum\_{i=1}^N \\hat{f}\_j(x\_{ij})  
\\end{align}  
$$

함수 $f\_j$가 미리 정해놓은 임계값 이하로 내려갈 때까지 반복합니다.

**설명을 해보자면 우선 (1)에서 $\\beta\_o$은 종속 변수 $y$의 평균값으로 초기화됩니다. $\\hat{f}\_j$ 또한 초기에는 0으로 초기화 됩니다.**

**(2)에서 각 변수 $j$에 대해 반복하는데, 주어진 식에따라 $\\hat{f}\_j$를 계산합니다. 다른 모든 변수의 부분적인 함수는 고정하고 주어진 변수 $j$에 대한 함수를 최적화화합니다. 이렇데 업데이트한 $\\hat{f}\_j$에 대해 추가적인 보정을 수행하고 이를 반복합니다.**

### ◾ In Classification

분류에서도 비슷합니다. 회귀에서 선형 모델을 기반으로 했다면 분류의 경우 호지스틱 회귀 모형을 기반으로 합니다.($\\log \\dfrac{P(Y=1|x\_i)}{1-P(Y=1|x\_i)} = \\beta\_0 + f\_1(x\_{i1}) + ... + \\beta\_p f\_p(x\_{ip})$ )

분류에서는 Local scoring Algorithm을 사용합니다.

$(1) \\text{ Compute starting values :} \\hat{\\beta}\_0 = \\log\[\\dfrac{\\bar{y}}{1-\\bar{y}}, \\text{ where } \\bar{y} = \\dfrac1N\\sum\_{i=1}^Ny\_i. \\text{ Set } \\hat{f}\_j \\equiv \\forall j \] \\\\ (2) \\text{ Define } \\hat{\\eta}\_i = \\hat{\\beta}\_0 + \\sum\_j \\hat{f}\_j (x\_{ij}) \\text{ and } \\hat{p}\_i = 1/\[1+ \\exp(-\\hat{\\eta}\_i)\]$

$$  
\\begin{align}  
&\\text{ Iterate :}  
\\\\ \\text{ (a)} &\\text{Construct the working target variable} : z\_i = \\hat{\\eta}\_i = \\dfrac{(y\_i - \\hat{p}\_i)}{\\hat{p}\_i(1-\\hat{p}\_i)}  
\\\\ \\text{ (b)} &\\text{Construct weights } :w\_i = \\hat{p}  
\\\\ \\text{ (c)} &\\text{Fit an additive model to the target } z\_i \\text{ with weights } w\_i,  
\\\\ &\\text{using a weighted backfitting algorithm.}  
\\\\ &\\text{This give new estimates} \\hat{\\beta}\_0, \\hat{f}\_j \\forall j  
\\end{align}  
$$

$\\text{(3) Continue step 2. }$

이 또한 미리정해둔 임계값 까지 반복합니다.

**설명해보면 (1)에서 $\\hat{\\beta}\_0$은 $y$의 평균인 $\\bar{y}$를 이용하여 초기화합니다. $\\hat{f}\_j$도 0으로 초기화.**

**(2)에서 target 변수인 $z\_i$를 지정해줍니다. 또한 가중치 $w\_i = \\hat{p}$로 초기화합니다.**

**이 가중치를 고려하여 로지스틱 회귀모델을 조정하고 가중치를 사용한 Backflitting 알고리즘을 사용하여 새로운 추정치인 $\\hat{\\beta}\_0, \\hat{f}\_j$를 얻습니다. 이를 반복합니다.**

### ◾ 장단점

#### **장점**

비선형 모델링 : 표준적인 선형 회귀가 간과할 수 있는 비선형 관계를 자동으로 모델링할 수있게 해준다.

예측 향상 : 비선형 데이터에 있어서 당연이 더 정확한 예측을 가능케 할 수 있다.

개별 변수 효과 : 모델의 가법성(additive)로 인해 각 예측변수가 반응 변수에 미치는 영향을 볼 수 있다.

부드러운 표현 : 함수 $f\_j$의 부드러움은 자유도의 개념을 사용하여 쉽게 알고 적용할 수 있다.

#### **단점**

가산성의 제한 : 아이러니하게도 한계는 모델이 가산적이라는 점... 변수가 많을 수록 이에 대한 상호작용이 존재할 것이고, 이러한 상호작용은 모델에 적용하지 않았다. 일반적인 선형 모델처럼 상호작용항을 추가할 수 있긴하다.
