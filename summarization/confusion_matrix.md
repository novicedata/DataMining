## 📌 Confusion matrix

---

데이터 과학 및 머신 러닝 분야에서 모델의 성능을 평가하고 해석하는 데 있어서, **혼동 행렬(Confusion Matrix)은 핵심적인 개념 중 하나입니다.**

혼동 행렬은 모델이 예측한 결과와 실제 관측값 간의 관계를 시각적으로 정리하는 강력한 도구로, 모델의 성능을 이해하고 향상시키는 데 도움이 됩니다.

혼동 행렬은 분류 모델의 평가를 위한 핵심 도구로, 모델이 얼마나 정확하게 예측하는지, 어떤 유형의 오류가 발생하는지를 살펴볼 수 있는 표입니다.

**Positive 클래스와 Negatice 클래스 간의 참 양성(True Positive), 참 음성(True Negative), 거짓 양성(False Positive), 그리고 거짓 음성(False Negative)을 나타내는 네 가지 핵심 지표를 제공합니다. 이러한 정보는 모델의 강점과 약점을 파악하고, 향후 개선 사항을 도출하는 데 도움이 됩니다.**

아래와 같은 표로 확인하면 이해가 더욱 쉬울 것입니다.

![](file:///C:/Users/jin/AppData/Roaming/marktext/images/2023-12-05-17-20-39-image.png?msec=1701764443761)[##_Image|kage@cYGfKh/btsBqzTIVNP/dCFfOA8aoY9yjQvGZA31iK/img.png|CDM|1.3|{"originWidth":498,"originHeight":212,"style":"alignCenter"}_##]

**이에 대해 정확도, 민감도, 특이도 등 여러 성능 지표가 있는데 하나하나 알아봅시다.**

**전체 적인 표입니다. 추가 설명은 아래에 있으니 참고하시길 바랍니다.**

| 정확도(Accuracy) | $\\dfrac{TP + TN}{Whole}$ |
| --- | --- |
| 민감도(Sensitivity) | $ \\dfrac{TP}{TP+FN} $ |
| 특이도(Specificity) | $ \\dfrac{TN}{TN+FP} $ |
| PPV(Positive Predictive Value), 정밀도(Precision) | $ \\dfrac{TP}{TP+FP} $ |
| NPV(Negative Predictive Value), 재현율(Recall) | $ \\dfrac{TN}{TN+FN} $ |
| Balanced Accuracy | $ \\dfrac{\\text{Sensitivity + Specificity}}{2} $ |
| $F\_\\beta$ | $ F\_\\beta = \\dfrac{(1+\\beta^2) \\times \\text{Precision} \\times \\text{Recall}}{(\\beta^2 \\times \\text{Precision}) + \\text{Recall}} $ |

◾ **정확도(Accuracy)** : 정확도는 TP+FP+FN+TN인 전체 값대 옳은 예측을 했던 TP와 TN을 더한 비율.

$$  
\\text{Accuracy} = \\dfrac{TP + TN}{Whole}  
$$

**◾ 민감도(Sensitivity)** : 민감도는 실제로 긍정인 샘플을 긍정이라고 분류한 것과 실제 모든 긍정인 샘플의 비율.

$$  
\\text{Sensitivity} = \\dfrac{TP}{TP+FN}  
$$

**◾ 특이도(Specificity)** : 특이도는 민감도와 반대로 실제로 부정인 샘플을 부정이라고 분류한 것과 실제 모든 부정인 샘플의 비율

$$  
\\text{Specificity} = \\dfrac{TN}{TN+FP}  
$$

**◾ PPV(Positive Predictive Value)** : 실제로 긍정인 샘플을 긍정이라고 분류한 것과 모델이 긍정이라고 분류한 모든 샘플의 비율

$$  
\\text{PPV} = \\dfrac{TP}{TP+FP}  
$$

**◾ NPV(Negative Predictive Value)** : 실제로 부정인 샘플을 부정이라고 분류한 것과 모델이 부정이라고 분류한 모든 샘플의 비율

$$  
\\text{NPV} = \\dfrac{TN}{TN+FN}  
$$

**◾ Balanced Accuracy** : imbalanced data일 때 주로 보는 성능으로 민감도와 특이도의 평균으로 볼 수 있다.

$$  
\\text{Balanced Accuracy} = \\dfrac{\\text{Sensitivity + Specificity}}{2}  
$$

**◾ 정밀도(Precision)** : 긍정이라고 예측한 샘플 중 실제로 긍정인 비율(PPV와 같음)

$$  
\\text{Precision} = \\dfrac{TP}{TP+FP}  
$$

**◾ 재현율(Recall)** : 부정이라고 예측한 샘플 중 실제로 부정인 비율(NPV와 같음)

$$  
\\text{Recall} = \\dfrac{TN}{TN+FN}  
$$

**◾ $F\_\\beta$** : 정밀도와 재현율의 조화평균으로, imbalanced data에서 모델의 성능을 평가할 때 유용.

$$  
F\_\\beta = \\dfrac{(1+\\beta^2) \\times \\text{Precision} \\times \\text{Recall}}{(\\beta^2 \\times \\text{Precision}) + \\text{Recall}}  
$$
