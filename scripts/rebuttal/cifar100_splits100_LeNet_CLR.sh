
ORDER=A # A,B,C,D,E
# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/run_cifar100.py
DATADIR=${SOURCEDIR}/data
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
EPOCH=20
BATCHSIZE=128
CHECKEPOCH=0

# CLR
python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --order_type ${ORDER} --train_size ${TRAINSIZE}  --valid_size 0.2 \
 --optimizer Adam --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 0.001 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 1 \
 --agent_type CLR --agent_name CLR \
 --wd_rate 1e-4 \