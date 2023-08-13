# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/run_five.py
DATADIR=data
LOGDIR=logs
TABDIR=accTable
TUNEDIR=tune
folderName=five_ResNet18

# Setting
MODEL=ResNet18
NWORK=4
SEED=0
REPEAT=3
TRAINSIZE=1.0
EPOCH=100
BATCHSIZE=128
CHECKEPOCH=0

# PRD
python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --model ${MODEL} --train_size ${TRAINSIZE}  --valid_size 0.2 \
 --optimizer SGD --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 0.001 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 5 \
 --agent_type PRD --agent_name PRD \
 --reg_coef 0 \
 --wd_rate 1e-4 \
 --weight_decay 1e-4 \
 --supcon_temperature 0.1 \
 --hidden_dim 512 --feat_dim=128 \
 --distill_coef 4.0 --distill_temp 1 \
 --prototypes_coef 2.0 --prototypes_lr 0.001 \
 --num_layers 3