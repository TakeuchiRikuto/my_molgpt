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

# メモ
コードの動かし方
python preprocessing.py --debug --output moses2_debug.csv
python train/train.py --run_name debug_test --data_name moses2_debug --batch_size 32 --max_epochs 1 --num_props 0 --debug
python generate/generate.py --model_weight weights/debug_test.pt --data_name moses2_debug --csv_name moses_scaf_tpsa_temp0.1 --gen_size 1000 --batch_size 32

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


