# 2-6 일본 영수증 메뉴 분석
## 문제
- Glace CIC에서 운영하는 [LINE CONOMI](https://lineconomi.me/) 서비스는 영수증을 기반으로 맛집이나 나들이 장소를 비롯한 다양한 곳의 리뷰를 남기고 검색할 수 있는 서비스입니다. 
- 사용자가 방문한 장소에서 받은 종이 영수증의 사진을 찍어 등록하면 영수증의 정보는 OCR을 통해 분석됩니다. 
이 과제는 음식점 영수증에 한하여 OCR로 인식된 메뉴명에 대해 아래 11개 entity를 분석하여 메뉴명과 옵션 정보를 추출하는 문제입니다. 
- 한국어로 예를들면 `아이스아메리카노L `라는 메뉴명이 있으면 아래와 같이 추출합니다
    - 아이스 -> 온도
    - 아메리카노 -> 메뉴
    - L -> 사이즈
 - 추출하려는 entity는 아래와 같습니다
  
한글 개체명 | 일본어 개체명
-- | --
메뉴 | メニュー名
구성 | 構成
기타 | その他
수량 | 数量
사이즈 | サイズ
중량/용량 | 重量/容量
프로모션 | プロモーション
매장/포장 | 店内/テイクアウト
온도 | 温度
추천 | おすすめ
맛 | 味


## Dataset
 - train/train_data
   - 학습데이터는 영수증 메뉴명 OCR 인식 text에서 11개의 entity 영역을 추출한 데이터 입니다 일본어로 구성되어있습니다
   - 학습데이터 수
      - 219692개
   - dataset 예시
```json
{
  "text": "+【ドリンクセット】アイスティー", // + [음료세트] 아이스티
  "value": "アイス", // 아이스
  "entity": "温度", // 온도
  "value_start": 10,
  "value_end": 12
}
```


  - field 상세 설명
      - text : 메뉴명 OCR 인식 text
      - entity : 11개의 entity 중 하나의 값
      - value : text에서 entity에 해당하는 text
      - value_start : text에서 value가 시작하는 index
      - value_end : text에서 value가 끝나는 index

   - format
       - json


## Evaluation
 - 평가 metric은 micro-avg f1 score 평균 입니다

## Baseline
 - Baseline 코드의 모델 구조는 [Scaling up Open Tagging from Tens to Thousands: Comprehension Empowered Attribute Value Extraction from Product Title](https://www.aclweb.org/anthology/P19-1514/) 에서 소개된 구조로 구현하였습니다
 - Pretrained model은 [Clova larva](https://oss.navercorp.com/Conversation/LaRva) 에서 제공하는 `larva-jpn-plus-base-cased`을 사용하였으며 bert 모델이므로 적절하게 변경하여 성능을 올려주시면 될 것 같습니다

## 학습 및 평가 방법
 - 학습
```
nsml run -d airush2021-2-6 -e train.py -a "--batch-size 100" --gpu-model V100
```
 - 리더보드 제출
```
nsml submit -t {NSML 계정}/airush2021-2-6/{Session Number} {Checkpoint name}
```
