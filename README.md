# Anomaly_detection_modeling

---


+ 주의 사항

---
: 해당 모델은 broken_large data와 정상 data인 good data를 기반으로 binary classification을 실시하고 있으며 이를 위해서는 폴더 내에 있는 다른 data들을 모두 지우고 실행해야하며 train dataset에 broken_large data의 일부를 training 폴더로 옮겨주어야 한다.

+ 다른 dataset을 활용하거나 multiclass classification을 하고싶은 경우
---
: 해당 경우에는 코드 내에서 outlayer를 class의 숫자만큼 다시 설정해주고 training dataset에 위와 같이 이상치 data를 추가해준 후 실행시키면 된다.


