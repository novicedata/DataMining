## 📌 Naive Bayes Classifier

나이브 베이즈는 텍스트 분석 분야에서 아직도 사용되고 있는 것 같습니다.(제가 잘 몰라서 틀릴 수도 있습니다.) 스팸 판단 등에서 사용되는 것 같더군요. 나온지 오래된 모델임에도 충분히 경쟁력이 있는 것 같습니다.

나이브 베이즈 분류기는 간단한 기술입니다. 단일 알고리즘을 통한 훈련이 아닌 일반적 원칙에 근거하여 여러 알고리즘들을 이용하여 훈련합니다.

**나이브 베이즈 분류기의 큰 특징이라함은? 모든 feature들이 조건부로 독립이라는 가정입니다.**

예를 들어봅시다. 특정 동물을 호랑이라고 분류합니다. 줄무늬, 울음소리, 고양이과등과 같은 특성들을 서로 독립적으로 동물이 호랑이일 확률에 기여한다고 가정합니다.

그럼 중요한 문제는 과연 feature들이 조건부 독립이라는 것이 왜 중요한 가정일까??? 입니다. **이는 모델을 훨씬 간단하게 만드는데 기여하기 때문입니다.**

베이즈 이론에 따라 확률을 봅시다. 만약 $n$개의 feature가 있는 예측 변수 벡터 $\\textbf{x} = (x\_1,...,x\_n)$이 있다면 이에 따른 클래스 할당 확률을 베이즈 이론에 따라 표현하면 아래와 같습니다.

$$  
p(Y\_k | x\_1,...,x\_n) = \\dfrac{p(Y\_k)p(x\_1,...,x\_n|Y\_k)}{p(x\_1,...,x\_n)}  
$$

분모의 경우 필요없는 상수이기 때문에 비례식을 만들고 분자식은 아래의 좌변과 같이 결합확률로 나타낼수 있다. 이를 조건부 확률의 성질을 이용하여 표현하면 아래와 같다.

$$  
\\begin{align}  
p(Y\_k, x\_1,...,x\_n) &= p(Y\_k)p(x\_1,...,x\_n|Y\_k)  
\\\\ &= p(Y\_k)p(x\_1|Y\_k)p(x\_2,...,x\_n|Y\_k, x\_1)  
\\\\ &= p(Y\_k)p(x\_1|Y\_k)p(x\_2|Y\_k, x\_1)p(x\_3,...,x\_n|Y\_k,x\_1,x\_2)  
\\\\ &= ...  
\\end{align}  
$$

이때 조건부 독립성에 의해 아래와 같은 식이 성립가능하다.($i \\neq j,k$)

$$  
p(x\_i|Y\_k, x\_j) = p(x\_i|Y\_k)  
\\ p(x\_i|Y\_k, x\_j, x\_k) = p(x\_i|Y\_k)  
$$

그럼 이러한 성질을 통해 결합 모델을 아래와 같이 표현할 수 있게 됩니다.

$$  
P(\\textbf{x}|Y\_k) = P(x\_1,...,x\_n|Y\_k) = \\prod\_{i=1}^n P(x\_i|Y\_k)  
$$

**이렇게 우도를 차원별 확률의 곱으로 분해할 수 있기 때문에 조건부 독립성 가정이 중요한 가정인 것입니다.**

### ◼️ 가우시안 나이브 베이즈(연속 변수)

**양적 예측 변수를 처리 할 때, 가우시안 분포를 따른 다고 가정합니다.** 각 클래스에 따라 평균과 분산을 이용하여 정규 분포 밀도 함수를 이용하여 확률을 계산합니다.

$$  
\\hat{\\mu}\_{ki} = \\dfrac1{n\_k} \\sum\_{y\_k = K} x\_{ki}  
\\\\ \\hat{\\sigma^2}\_{ki} = \\dfrac1{n\_k} \\sum\_{y\_k=K} (x\_{ki}-\\hat{\\mu}{ki})^2  
$$  
$$  
P(\\textbf{x}|Y\_k) = \\prod\_{i=1}^n \\dfrac1{\\sqrt{2\\pi}\\sigma\_{ki}}\\exp\\left\[-\\dfrac{(x\_i -\\mu\_{ki})^2}{2\\sigma^2\_{ki}} \\right\]  
$$

### ◼️ 다항 나이브 베이즈(범주형 변수)

**질적 예측 변수(categorial)를 처리 할 때는 다항 분포를 이용하여 계산합니다.**특정 클래스에서 $\\textbf{x}$가 특정 범주일 확률이 다항 분포에 따른다고 가정하는 것입니다. 우도는 아래와 같습니다.

$$  
p(\\textbf{x}|Y\_k) = \\dfrac{(\\sum\_i x\_i)!}{\\prod\_i x\_i !}\\prod p\_{ki}^{x\_i}  
$$

이를 로그 공간에서 표현하면 아래와 같습니다.

$$  
\\begin{align}  
\\log p(Y\_K|\\textbf{x}) &\\propto \\log \\left(p(Y\_k) \\prod\_{i=1}^n p\_{ki}^{x\_i} \\right)  
\\\\ & \\propto \\log p(Y\_k) + \\sum\_{i=1}^n x\_i \\cdot \\log p\_{ki}  
\\\\ & \\propto b+ \\textbf{w}^T\_k \\textbf{x}  
\\end{align}  
$$

이렇게 어떠한 가중치를 가진 $x$벡터의 식으로 표현할 수 있습니다.

### ◼️ 베르누이 나이브 베이즈(범주형 변수)

질적 예측 변수인데 이진 변수인 경우입니다. 쉽기 때문에 식만 놓고 넘어가겠습니다. 베르누이 분포를 따른다고 가정합니다.

$$  
p(\\textbf{x}|Y\_k) = \\prod\_{i=1}^n p\_{ki}^{x\_i} (1-p\_{ki})^{1-x\_i}  
$$

**재밌는 점은 연속 변수인 경우에도 이를 범주형으로 바꾸어서 다항 나이브 베이즈를 적용할 수 있다는 점입니다.** 정보를 잃긴 하지만 범주형으로 바꾸어도 크게 상관이 없고, 분류 자체에 목적이 크다면 나쁘지 않은 접근법일 수도 있겠습니다.

예를 들어, 대학의 선형대수학 강의 중간고사+기말고사 점수를 0 ~ 50점으로 보았을 때 이를 F ~ A 로 바꾸는 것입니다.