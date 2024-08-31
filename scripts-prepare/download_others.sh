set -e  # return if any errors

DATA_DIR="data"
if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR}
fi

python scripts-prepare/download_others.py --data_root ${DATA_DIR}