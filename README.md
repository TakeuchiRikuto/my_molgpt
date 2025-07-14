# MolGPT
In this work, we train small custom GPT on Moses and Guacamol dataset with next token prediction task. The model is then used for unconditional and conditional molecular generation. We compare our model with previous approaches on the Moses and Guacamol datasets. Saliency maps are obtained for interpretability using Ecco library.

- The processed Guacamol and MOSES datasets in csv format can be downloaded from this link:

https://drive.google.com/drive/folders/1LrtGru7Srj_62WMR4Zcfs7xJ3GZr9N4E?usp=sharing

- Original Guacamol dataset can be found here:

https://github.com/BenevolentAI/guacamol

- Original Moses dataset can be found here:

https://github.com/molecularsets/moses

- All trained weights can be found here:

https://www.kaggle.com/virajbagal/ligflow-final-weights


To train the model, make sure you have the datasets' csv file in the same directory as the code files.

Environment Setup Guide
macOS (M1/M2/M3/M4) Setup
bash# 1. Create conda environment
conda create -n molgpt python=3.12 -y
conda activate molgpt

# 2. Install conda packages (recommended)
conda install -c conda-forge rdkit pandas numpy matplotlib seaborn tqdm -y

# 3. Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio

# 4. Install remaining packages
pip install PyTDC wandb
Linux with CUDA Setup
bash# 1. Create conda environment
conda create -n molgpt python=3.12 -y
conda activate molgpt

# 2. Install conda packages
conda install -c conda-forge rdkit pandas numpy matplotlib seaborn tqdm -y

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining packages
pip install PyTDC wandb
Linux CPU-only Setup
bash# 1. Create conda environment
conda create -n molgpt python=3.12 -y
conda activate molgpt

# 2. Install conda packages
conda install -c conda-forge rdkit pandas numpy matplotlib seaborn tqdm -y

# 3. Install PyTorch CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining packages
pip install PyTDC wandb
Simple requirements.txt (fallback)
For pip-only installation (may have issues with RDKit):
pandas
numpy
matplotlib
seaborn
tqdm
PyTDC
wandb
torch
Notes

RDKit: Always install via conda when possible
PyTorch: Version depends on your hardware (CPU/CUDA/Apple Silicon)
argparse: Built into Python 3.12
Test your installation with: python -c "import rdkit; import torch; print('Success!')"


# メモ
コードの動かし方
python preprocessing.py --debug --output moses2_debug.csv##これはしないcsvを別途ダウンロード



python train/train.py --run_name  mose2_debug --data_name moses2_debug --batch_size 32 --max_epochs 1 --num_props 0 --debug
python generate/generate.py --model_weight weights/debug_test.pt --data_name moses2_debug --csv_name moses_scaf_tpsa_temp0.1 --gen_size 1000 --batch_size 32

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121pip install PyTDC wandb

python generate/generate.py \
  --model_weight weights/debug_test.pt \
  --data_name moses2_debug \
  --csv_name test_generation \
  --gen_size 100 \
  --batch_size 32 \
  --vocab_size 94 \
  --block_size 49 \
  --debug

# Training

```
./train_moses.sh
```

```
./train_guacamol.sh
```

# Generation

```
./generate_guacamol_prop.sh
```

```
./generate_moses_prop_scaf.sh
```

If you find this work useful, please cite:

Bagal, Viraj; Aggarwal, Rishal; Vinod, P. K.; Priyakumar, U. Deva (2021): MolGPT: Molecular Generation using a Transformer-Decoder Model. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.14561901.v1 


