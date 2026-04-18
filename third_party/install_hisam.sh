# Ensure we are running from the third_party directory
if [ "$(basename "$PWD")" != "third_party" ]; then
    echo "Error: Please run this script from the third_party directory."
    exit 1
fi

if [ -d "Hi-SAM" ]; then
    echo "Hi-SAM repository already exists, skipping cloning."
else
    echo "cloning Hi-SAM repository (forked)..."
    git clone git@github.com:GeoRAMIEL/Hi-SAM.git
    echo "Hi-SAM repository cloned. Attention: no need to install Hi-SAM dependencies — inference deps overlap with the main .venv (torch, torchvision, einops, shapely, pyclipper, scikit-image, scipy, matplotlib, pillow, tqdm, opencv, numpy)."
fi

mkdir -p Hi-SAM/pretrained_checkpoint

# ---------------------------------------------------------------------------
# SAM ViT-L encoder weights (wget from Meta)
# Used by Hi-SAM's build.py to initialize the frozen ViT image encoder.
# ---------------------------------------------------------------------------
if [ -f "Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth" ]; then
    echo "SAM ViT-L encoder checkpoint exists, skipping download."
else
    if ! command -v wget &> /dev/null
    then
        echo "wget could not be found, please install it to download the SAM ViT-L encoder checkpoint"
        echo "Alternatively, you can download the checkpoint manually from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth and place it in Hi-SAM/pretrained_checkpoint/"
        exit
    fi
    echo "Downloading SAM ViT-L encoder checkpoint..."
    wget -O Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
fi

# ---------------------------------------------------------------------------
# Hi-SAM SAM-TS-L TextSeg head weights (gdown from Google Drive)
# Stroke-level text segmentation head trained on TextSeg.
# ---------------------------------------------------------------------------
if [ -f "Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth" ]; then
    echo "Hi-SAM SAM-TS-L TextSeg checkpoint exists, skipping download."
else
    if ! command -v gdown &> /dev/null
    then
        echo "gdown could not be found, please install it to download the Hi-SAM checkpoint"
        echo "You can install it with pip: pip install gdown"
        echo "Alternatively, you can download the checkpoint manually from https://drive.google.com/file/d/1aIPUic7Q0dz3dXhLSYa0GH4t8EwO2FXG/view and place it in Hi-SAM/pretrained_checkpoint/"
        exit
    fi
    echo "Downloading Hi-SAM SAM-TS-L TextSeg checkpoint..."
    gdown https://drive.google.com/uc?id=1aIPUic7Q0dz3dXhLSYa0GH4t8EwO2FXG -O Hi-SAM/pretrained_checkpoint/sam_tss_l_textseg.pth
fi
