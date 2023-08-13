# Directory
SOURCEDIR=.
MAINFILE=${SOURCEDIR}/run_five.py
DATADIR=data
LOGDIR=logs
TABDIR=accTable
TUNEDIR=tune
folderName=five_ResNet18

# Setting
MODEL=ResNet18 # "AlexNet" or "ResNet18"
NWORK=2
SEED=0
REPEAT=5
TRAINSIZE=1.0
EPOCH=12
BATCHSIZE=128
CHECKEPOCH=0

# FHALRP
python -u ${MAINFILE} --workers ${NWORK} --folderName ${folderName} --source_dir ${SOURCEDIR} --data_dir  ${DATADIR} --log_dir ${LOGDIR}   --tab_dir ${TABDIR} --tune_dir ${TUNEDIR} \
 --seed ${SEED} --repeat ${REPEAT} --model ${MODEL} --train_size ${TRAINSIZE}  --valid_size 0.2 \
 --optimizer Adam --n_epochs  ${EPOCH} --batch_size ${BATCHSIZE}   --lr 0.001 --check_lepoch ${CHECKEPOCH} \
 --print_freq 200 --epoch_freq 1 \
 --agent_type HALRP_Res --agent_name HALRP_Res \
 --wd_rate 1e-4 \
 --reg_coef 0 \
 --l1_hyp 1e-6 \
 --approxiRate 0.95 --upper_rank 120 --estRank_epoch 3 \
 --prune_method mixAR_AllLayer  --prune_value 1e-5 0.40 --prune_boundAlltask \
 --method_desc p1e5_b60_a95_up120_er3