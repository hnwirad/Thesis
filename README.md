# Hairstyle Transfer Experiment

Based on: "Realism Enhancement Techniques for Hair Style Transfer Using Generative Adversarial Network (GAN)"

## Struktur Project
- requirements.txt
- dataset_prep.py
- models/mine.py
- losses.py
- train.py
- evaluate.py
- inference.py
- utils.py

## Cara Jalanin di VS Code
1. Buat virtualenv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Preprocessing dataset:
   ```bash
   python dataset_prep.py --input_dir data/images --out_dir data/processed --size 1024
   ```

4. Training (scaffold):
   ```bash
   python train.py --data_dir data/processed --epochs 200
   ```

5. Evaluasi:
   ```bash
   python evaluate.py --real_dir data/processed --fake_dir results/generated
   ```

6. Inference (scaffold):
   ```bash
   python inference.py --source source.jpg --target target.jpg --out result.jpg
   ```
