#!/bin/bash

# Install PaddlePaddle (https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
python -m pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# Install PaddleOCR toolkit (basic text recognition feature only)
python -m pip install paddleocr

# if on a headless server, uninstall opencv-python, install opencv-python-headless instead to avoid GUI-related errors
# python -m pip uninstall -y opencv-python
# python -m pip install opencv-python-headless
