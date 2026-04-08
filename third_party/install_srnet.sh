# Ensure we are running from the third_party directory
if [ "$(basename "$PWD")" != "third_party" ]; then
    echo "Error: Please run this script from the third_party directory."
    exit 1
fi

if [ -d "SRNet" ]; then
    echo "SRNet repository already exists, skipping cloning."
else
    echo "cloning SRNet repository..."
    git clone https://github.com/lksshw/SRNet.git
    echo "SRNet repository cloned. Attention: no need to install SRNet dependencies since we only use a part of it."
fi

if [ -f "SRNet/checkpoints/trained_final_5M_.model" ]; then
    echo "SRNet checkpoint exists, skipping download."
else
    if ! command -v gdown &> /dev/null
    then
        echo "gdown could not be found, please install it to download the SRNet checkpoint"
        echo "You can install it with pip: pip install gdown"
        echo "Alternatively, you can download the checkpoint manually from https://drive.google.com/file/d/1FMyabJ5ivT3HVUfUeozqOpMqlU68V65K/view and place it in SRNet/checkpoints/"
        exit
    fi
    mkdir SRNet/checkpoints
    echo "Downloading SRNet checkpoint..."
    gdown https://drive.google.com/uc?id=1FMyabJ5ivT3HVUfUeozqOpMqlU68V65K -O SRNet/checkpoints/trained_final_5M_.model
fi
