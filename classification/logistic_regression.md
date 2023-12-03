## 📌 로지스틱 회귀

---

로지스틱 회귀에 대한 간단한 식의 설명. 계수 구하는 방법과 같은 과정은 후에 다뤄보도록 하자.

일반적인 선형회귀에 대하 아실것이라 가정하고 로지스틱 회귀를 한번 봅시다. 기본적으로 독립 변수들의 선형 결합으로 표현하는 방정식을 통해 종속 변수를 표현하는 것을 유사 합니다.

**가장 큰 차이는 종속 변수의 차이입니다. 선형 회귀의 경우 연속 변수이지만 로지스틱 회귀의 경우 종속 변수가 범주형 변수일 경우 사용하게 됩니다.** 그러니 이름에 회귀!가 있지만? 일종의 분류 기법으로 사용하는 것입니다.

**정말 간단한 모델로 우리가 어떠한 분류 문제를 다룰 때 제일 먼저 사용해보아야할 모델**이기도 합니다. 간단한 모델로 충분한 성능이 나온다면 굳이 어렵고 복잡한 모델을 사용할 필요가 적겠죠??

**회귀의 결과값은 아래와 같이 항상 \[0,1\]로 제한되어 있습니다. 이미 포스팅 했던 분류에 회귀를 사용하지 않는 이유를 보시면 이해가 편할 것입니다.** ([링크 : 분류에 선형회귀를 사용하지 않는 이유](https://datanovice.tistory.com/entry/%EB%B6%84%EB%A5%98%EC%97%90-%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%98%EC%A7%80-%EC%95%8A%EB%8A%94-%EC%9D%B4%EC%9C%A0))

![](file://C:%5CUsers%5Catlsw%5CAppData%5CRoaming%5Cmarktext%5Cimages%5C2023-12-03-17-05-22-image.png?msec=1701590722571)[##_Image|kage@bSnxt1/btsBiRz1IwV/93LTVJzAdynw6HJKFytxk1/img.png|CDM|1.3|{"originWidth":647,"originHeight":482,"style":"alignCenter","width":493,"height":367}_##]

로지스틱을 이용하면 더이상 값이 0 미만, 1 초과로 이어지지 않기 때문에 분류에 사용할 수 있는 것입니다.

### ◼️ odds, logit

$Y= \\text{1 or 0}$으로 봅시다. 이에 대한 로지스틱 회귀는 reponse가 특정 범주에 속할 확률 $P(Y=1|X)$를 모델링합니다.

성공확률(여기선 1을 성공으로 보겠습니다.)이 실패 확률에 비해 얼마나 높은가? 를 나타내는 odds(오즈) 식은 아래와 같습니다.

$$  
odds = \\dfrac{P(Y=1|X)}{P(Y=0|X)} = \\dfrac{P(Y=1|X)}{1-P(Y=1|X)}  
$$

이 odds에 로그를 취하여 값의 범위를 \[0, 1\] 로 제한하는 것이 아이디어 입니다.

$$  
logit = \\log(odds) = \\log \\left( \\dfrac{P(Y=1|X)}{1-P(Y=1|X)} \\right)  
$$

이를 회귀식으로 표현하여 아래와 같은 로지스틱 회귀식을 만들고 각 베타값을 찾습니다.

$$  
\\log \\left( \\dfrac{P(Y=1|X)}{1-P(Y=1|X)} \\right) = \\beta\_0 + \\beta\_1 X  
$$

이 때 $p(Y=1|X)$를 간단하게 $p(X)$로 표현하고 $p(X)$에 대해 정의하면 아래와 같습니다.

$$  
P(Y=1|X) = p(X) = \\dfrac{\\exp(\\beta\_0 + \\beta\_1 X)}{1 + \\exp(\\beta\_0 + \\beta\_1 X)}  
\\ P(Y=0|X) = 1-p(X) = \\dfrac{1}{1 + \\exp(\\beta\_0 + \\beta\_1 X)}  
$$

### ◼️ 로지스틱 회귀에 대한 해석

**선형회귀에서 $\\beta\_0$은 intercept이고 $\\beta\_1$은 기울기 인데 로지스틱 회귀에서 $\\beta\_1$은 무엇을 뜻할까? (당연 $\\beta\_0$은 X가 0일 때의 값일 것이다.)**

$X+1$에서의 로짓에서 $X$에서의 로짓을 빼서 $\\beta\_1$을 확인하면 아래와 같습니다.

$$  
\\log\\dfrac{p(X+1)}{1-p(X+1)} - \\log\\dfrac{p(X)}{1-p(X)} = \\beta\_0+\\beta\_1(X+1) - (\\beta\_0 + \\beta\_1X)  
$$

$$  
\\log \\dfrac{p(X+1)(1-p(X))}{(1-p(X+1))p(X)} = \\beta\_1  
$$

**위 식으로 보았을 때 이는 클래스 $Y=1$의 로짓($\\log(odds)$)의 $X+1$과 $X$간의 비율입니다. 쉽게 설명하면 1이 더해졌을 때 증가하는 odds의 비율을 알 수 있다는 것입니다.**

예를 들어 $\\beta\_1 = 0.4$ 라면 $odds$를 구하기 위해 exponential을 취하면 $\\exp(0.4) \\approx 1.5$가 됩니다. 이는 $X+1$에서의 $odds$가 $X$에서의 $odds$보다 1.5배 라는 것이겠죠.

## 📌다중 로지스틱 회귀(Multinomial logistic regression)

---

단순히 0 or 1, Yes or No의 2가지 클래스가 아닌 여러 클래스가 있을 경우에는 로짓 모델을 아래와 같이 표현할 수 있습니다. 행렬로 표현이 가능합니다.

$$  
\\begin{align}  
\\log \\dfrac{P(Y=i|X\_1=x\_1,...,X\_p=x\_p)}{P(Y=K|X\_1=x\_1,...,X\_p=x\_p)} &= \\beta\_{i0} + \\beta\_{i1}x\_1 +...+ \\beta\_{ip}x\_p \\  \\ (i = 1,..., K-1) \\\\ &= \\pmb{\\beta\_i}^T \\textbf{x}  
\\end{align}  
$$

이를 통해 $\\textbf{X} = \\textbf{x}$가 주어졌을 때 $Y=i$의 조건부 확률을 구하면 아래와 같습니다.

$$  
P(Y=i|\\textbf{X} = \\textbf{x}) = \\dfrac{\\exp(\\pmb{\\beta\_i}^T \\textbf{x})}{1+ \\Sigma\_{m=1}^{K-1} \\exp(\\pmb{\\beta\_m}^T \\textbf{x})}  
\\\\ P(Y=K|\\textbf{X} = \\textbf{x}) = \\dfrac{1}{1+ \\Sigma\_{m=1}^{K-1} \\exp(\\pmb{\\beta\_m}^T \\textbf{x})}  
$$

**좀 더 복잡해 보일뿐, 기본적인 원리와 구조는 이진 로지스틱 회귀와 유사한걸 알 수 있습니다.**

class 3개일 때의 분류 그래프

[##_Image|kage@cFqLAB/btsBhXABWN5/oJYI5ChmOmGm1JKkTkXPW1/img.png|CDM|1.3|{"originWidth":677,"originHeight":652,"style":"alignCenter","width":540,"height":520}_##]
