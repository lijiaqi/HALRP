
ORDER=$1 # A,B,C,D,E
# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/run_cifar100.py
DATADIR=data
LOGDIR=logs
TABDIR=accTable
TUNEDIR=tune
folderName=cifar100_LeNet_splits100

# Setting
TRAINSIZE=1.0
MODEL=LeNet 
DATATYPE=default
NWORK=4
SEED=111111
REPEAT=3
EPOCH=100
BATCHSIZE=128
CHECKEPOCH=0

# PRD
python -u ${MAINFILE} --model ${MODEL} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --order_type ${ORDER} --train_size ${TRAINSIZE}  --valid_size 0.2 \
 --optimizer SGD --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 0.001 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 5 \
 --agent_type PRD --agent_name PRD \
 --wd_rate 1e-4 \
 --weight_decay 1e-4 \
 --supcon_temperature 0.1 \
 --hidden_dim 512 --feat_dim=128 \
 --distill_coef 4.0 --distill_temp 1 \
 --prototypes_coef 2.0 --prototypes_lr 0.001 \
 --num_layers 3