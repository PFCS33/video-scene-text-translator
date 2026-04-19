# Download BPN checkpoint
# the checkpoint is on google drive, so we use gdown to download it
# install gdown if not already installed
# alternatively, you can download the checkpoint manually from https://drive.google.com/file/d/1ZUDMCDw6tJka-0Dxkhev2bRvkkLMcKpv/view?usp=drive_link
# and place it in checkpoints/bpn/

# check if gdown is installed, if not, print a message
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, please install it to download the checkpoint"
    echo "You can install it with pip: pip install gdown"
    echo "Alternatively, you can download the checkpoint manually from https://drive.google.com/file/d/1ZUDMCDw6tJka-0Dxkhev2bRvkkLMcKpv/view?usp=drive_link and place it in checkpoints/bpn/"
    exit
else
    # check if we are in the root directory of the project, if not, print a message
    if [ ! -d "code" ]; then
        echo "Please run this script from the root directory of the project"
        exit
    fi
    # create checkpoints/bpn directory if it doesn't exist
    mkdir -p checkpoints/bpn
    # check if bpn/bpn_v1.pt already exists, if it does, print a message and skip downloading
    if [ -f "checkpoints/bpn/bpn_v1.pt" ]; then
        echo "bpn_v1 checkpoint already exists, skipping download"
    else
        # download the checkpoint using gdown
        gdown https://drive.google.com/uc?id=1THY1y9Cwynvpz19Es_awI1UQIp07ppkS -O checkpoints/bpn/bpn_v1.pt
    fi
    mkdir -p checkpoints/refiner
    # check if refiner/refiner_v0.pt already exists, if it does, print a message and skip downloading
    if [ -f "checkpoints/refiner/refiner_v0.pt" ]; then
        echo "refiner_v0 checkpoint already exists, skipping download"
    else
        # download the checkpoint using gdown
        gdown https://drive.google.com/uc?id=1_6Fk5Q6Gg7OwL2ILe5ikYVhSd7JEBSRo -O checkpoints/refiner/refiner_v0.pt
    fi
    # check if refiner/refiner_v1.pt already exists, if it does, print a message and skip downloading
    if [ -f "checkpoints/refiner/refiner_v1.pt" ]; then
        echo "refiner_v1 checkpoint already exists, skipping download"
    else
        # download the checkpoint using gdown
        gdown https://drive.google.com/uc?id=1s_nY4L2FVdAv3Vh-l8JpJhFMKzNPLUFe -O checkpoints/refiner/refiner_v1.pt
    fi
fi
