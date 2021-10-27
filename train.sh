nsml run \
  -d airush2021-2-6a\
  -g 1 \
  -c 8 \
  -e train.py \
  --memory=32G \
  --shm-size 8G\
  -a "--amp --ema --model LSTM --optimizer AdamW --lr 5e-6 "
