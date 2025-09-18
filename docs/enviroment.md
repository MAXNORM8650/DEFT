# Create enviroment
## New enviroment
```bash
conda create -n deft python=3.10.13
conda activate deft
```
## Install torch with cuda
```bash
pip install torch==2.3.1+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```
### We use [omnigen](https://github.com/VectorSpaceLab/OmniGen.git) and diffusion models from [Dreambench_plus](https://github.com/yuangpeng/dreambench_plus) as baselines. To setup Omnigne,

```bash
git clone https://github.com/VectorSpaceLab/OmniGen.git
cd OmniGen
pip install -e .
```
### Also install dreambench_plus but new envirment is recommnaded and it will be used for evlaution
```bash
git clone https://github.com/yuangpeng/dreambench_plus.git
cd dreambench_plus
pip install -e .
```

## Repo QnA
### Omnigen uses phi1.5, so to use phi1.5 correctly, TypeError: cannot unpack non-iterable NoneType object
```bash
python -m pip install transformers==4.45.2
```
### To use Blip2, upgarde
```bash
python -m pip install --upgrade git+https://github.com/huggingface/transformers.git
```
