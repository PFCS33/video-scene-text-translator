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
    # download the checkpoint using gdown
    gdown https://drive.google.com/uc?id=1ZUDMCDw6tJka-0Dxkhev2bRvkkLMcKpv -O checkpoints/bpn/bpn_v0.pth
fi
