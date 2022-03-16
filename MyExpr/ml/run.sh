
# todo 还有数据集选择
LR=$1
MODEL=$2
EPOCH=$3
BS=$4
DEVICE=$5
NAME=${MODEL}-lr${LR}-bc${BS}

echo $NAME

python3 resnet_train.py \
--lr $LR \
--model $MODEL \
--epoch $EPOCH \
--batch_size $BS \
--device $DEVICE \
--name $NAME