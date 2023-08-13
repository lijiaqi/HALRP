# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/run_tiny.py
DATADIR=data
LOGDIR=logs
TABDIR=accTable
TUNEDIR=tune
folderName=tinyimagenet_ResNet18

# Setting
MODEL=ResNet18
NWORK=8
SEED=0
REPEAT=3
TRAINSIZE=1.0
EPOCH=100
BATCHSIZE=128
CHECKEPOCH=0
NUM_TASKS=20

#  PRD
python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --model ${MODEL} --train_size ${TRAINSIZE}  --valid_size 0.1 \
 --optimizer SGD --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 1e-3 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 5 \
 --agent_type PRD --agent_name PRD \
 --wd_rate 1e-4  \
 --weight_decay 1e-4 \
 --num_tasks ${NUM_TASKS} \
 --supcon_temperature 0.1 \
 --hidden_dim 512 --feat_dim=128 \
 --distill_coef 4.0 --distill_temp 1 \
 --prototypes_coef 2.0 --prototypes_lr 1e-3 \
 --num_layers 3