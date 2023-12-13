뉴럴네트워크에서 또한 정규화가 이루어집니다. 추가로 과적합을 피하기 위한 다른 방법인 드롭아웃에 대해 알아보겠습니다.

## 📌 정규화(Regularization)

---

정규화를 통해 모델 복잡성을 제어하여 과적합을 방지하고자 하는 목적입니다.

어느 모델과 마찬가지로 **L1, L2 Regularization**이 존재합니다. 손실 함수에 모델 파라미터의 크기에 대한 페널티를 추가하는 방식입니다.

**L1 정규화는 가중치의 절대값에 비례하는 페널티를 더하고,**

**L2 정규화는 가중치의 제곱에 비례하는 페널티를 더합니다.**

이를 통해 모델이 특정 가중치에 지나치게 의존하지 않도록 제한합니다.

$$  
\\text{cost function = loss + Reguralization term}  
$$

**정규화 항이 추가되면 가중치 행렬 값이 감소하게 되는데, 이는 작은 가중치 행렬을 가진 신경망이 더 단순한 모델을 의미한다고 가정하기 때문. 즉, 정규화 항을 추가하여 overfitting 되지 않도록 덜 복잡한 모델을 만드는 것입니다.**

### ◼️ L2

$$  
\\text{cost function = loss +} \\lambda||\\textbf{w}||\_2^2  
$$

여기서 $\\lambda$가 정규화 파라미터입니다. 라쏘와 릿지에 대해 배우신 분이라면 예측하시겠지만 L2 정규화는 가중치를 완전히 0이 되게 만들지는 않습니다. 단 0과 가깝게는 만들 수 있습니다.

### ◼️ L1

$$  
\\text{cost function = loss +} \\lambda||\\textbf{w}||\_1  
$$

이 또한 $\\lambda$가 하이퍼 파라미터이고, 절댓값을 이용하여 페널티를 부과하기 때문에 가중치가 0이 될 수도 있습니다. 따라서 불필요한 것을 0으로 만들어 모델을 압축하는 것이 가능합니다.

## 📌 드롭아웃

---

**학습 중에 무작위로 선택된 뉴런들을 제거(drop out)하여 모델이 각 뉴런에 과도하게 의존하지 않도록 하는 방법**입니다. 이를 통해 유닛 간의 상호의존성을 줄입니다.

**예를 들어 드롭아웃을 0.5로 설정하면? 각 훈련 단계에서 무작위로 뉴런이 0.5의 확률로 제거될지 말지 결정되게 됩니다. 이는 네트워크가 다양한 부분집합에서 학습하도록 만들어 일반화 성능을 향상합니다.**

![](file://C:%5CUsers%5Catlsw%5CAppData%5CRoaming%5Cmarktext%5Cimages%5C2023-12-13-20-31-09-image.png?msec=1702467069765)[##_Image|kage@kZbZp/btsBTWuzSAa/RbA9it0HLaOjWHfPw86B2K/img.png|CDM|1.3|{"originWidth":468,"originHeight":238,"style":"alignCenter","caption":"출처 : Dropout:A Simple Way to Prevent Neural Networks from Overfitting"}_##]

훈련 단계:

각 훈련 단계에서 개별 노드는 확률 $p$ 로 네트워크에서 제외되거나 유지됩니다. 이로 인해 축소된 구조의 네트워크가 남고, 제외된 노드에 연결된 입출력 edge도 제거됩니다.

테스트 단계 :

테스트 단계에서는 모든 노드 활성화를 사용하지만 각 활성화는 훈련 중 누락된 활성화를 고려하여 출력을 작게 줄여서(평균적으로 제외된 노드만큼) 정보를 보정합니다.

드롭아웃은 보통 수렴에 필요한 반복 횟수를 두 배로 만듭니다. 대신 각 에포크의 훈련 시간은 감소되겠죠?

**생각해 보면 머신러닝의 앙상블 모델과 비슷합니다. 설정된 확률로 신경망의 여러 부분 집합을 이용하여 예측을 하는 것이니까요.**