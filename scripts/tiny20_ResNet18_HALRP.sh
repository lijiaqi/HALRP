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
NWORK=4
SEED=0
REPEAT=3
TRAINSIZE=1.0
EPOCH=50
BATCHSIZE=32
CHECKEPOCH=0
NUM_TASKS=20


python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --model ${MODEL} --train_size ${TRAINSIZE}  --valid_size 0.1 \
 --optimizer Adam --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 5e-4 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 1 \
 --agent_type HALRP_Res --agent_name HALRP_Res \
 --wd_rate 1e-4  \
 --reg_coef 0 \
 --l1_hyp 5e-4 \
 --approxiRate 0.9 --upper_rank 100 --estRank_epoch 25 \
 --prune_method mixAR_AllLayer  --prune_value 1e-5 0.40 --prune_boundAlltask \
 --method_desc p1e5_b60_a90_up100_er25 \
  --num_tasks ${NUM_TASKS}
