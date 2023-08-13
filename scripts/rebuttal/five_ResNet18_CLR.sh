
# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/run_five.py
DATADIR=$SOURCEDIR/data
LOGDIR=logs
TABDIR=accTable
TUNEDIR=tune
folderName=five_ResNet18_100

# Setting
MODEL=ResNet18
NWORK=4
SEED=0
REPEAT=3
TRAINSIZE=1.0
EPOCH=12
BATCHSIZE=128
CHECKEPOCH=0

# CLR
python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --model ${MODEL} --train_size ${TRAINSIZE}  --valid_size 0.2 \
 --optimizer Adam --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 0.001 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 1 \
 --agent_type CLR --agent_name CLR \
 --reg_coef 0 --wd_rate 1e-4 \