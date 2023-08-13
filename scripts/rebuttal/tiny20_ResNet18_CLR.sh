
# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/tinyimagenet_run.py
DATADIR=$SOURCEDIR/data
LOGDIR=logs-tiny20
TABDIR=accTable-tiny20
TUNEDIR=tune-tiny20
folderName=tinyimagenet_ResNet18

# Setting
MODEL=ResNet18
NWORK=4
SEED=0
REPEAT=3
TRAINSIZE=1.0 # 用100%的数据量.
EPOCH=50
BATCHSIZE=32
CHECKEPOCH=0
NUM_TASKS=20


# CLR
python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --model ${MODEL} --train_size ${TRAINSIZE}  --valid_size 0.1 \
 --optimizer Adam --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 1e-3 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 1 \
 --agent_type CLR --agent_name CLR \
 --wd_rate 1e-4  \
 --num_tasks ${NUM_TASKS}
