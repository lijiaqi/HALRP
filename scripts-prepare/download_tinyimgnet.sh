set -e  # return if any errors

DATA_DIR="data"
if [ ! -d ${DATA_DIR} ]; then
    mkdir ${DATA_DIR}
fi

echo "Downloading Data ..."
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip -O ${DATA_DIR}/tiny-imagenet-200.zip
echo "Unzipping Data ..."
unzip -q ${DATA_DIR}/tiny-imagenet-200.zip -d ${DATA_DIR}/

echo "Deleting images in '${DATA_DIR}/tiny-imagenet-200/test' folder ..."
rm -r ${DATA_DIR}/tiny-imagenet-200/test/*

echo "Re-organizing '${DATA_DIR}/tiny-imagenet-200/val' folder ..."
python3 scripts-prepare/val_data_format.py ${DATA_DIR}
# find . -name "*.txt" -delete
# rm tiny-imagenet-200.zip

