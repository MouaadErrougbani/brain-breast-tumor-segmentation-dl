# brain-breast-tumor-segmentation-dl
Deep learning project for automated tumor segmentation in brain MRI and breast ultrasound images using U-Net, U-Net++, and DeepLabV3 with a ResNet34 backbone. Includes preprocessing, data augmentation, model training, evaluation, and performance comparison using Dice, Precision, Recall, F1-score, and Accuracy.

```
brain-breast-tumor-segmentation-dl/
│
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── Makefile
│
├── data/
│   ├── raw/                    # Données originales téléchargées
│   ├── interim/                # Données intermédiaires
│   ├── processed/              # Données prétraitées
│   └── splits/
│       ├── train/
│       ├── val/
│       └── test/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_analysis.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_experiments.ipynb
│   └── 05_evaluation.ipynb
│
├── ml/
│   ├── config/
│   │   ├── config.yaml
│   │   ├── train.yaml
│   │   └── model.yaml
│   │
│   ├── data/
│   │   ├── download.py
│   │   ├── dataset.py
│   │   ├── split.py
│   │   └── validation.py
│   │
│   ├── preprocessing/
│   │   ├── resize.py
│   │   ├── normalize.py
│   │   ├── augmentation.py
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   ├── unet.py
│   │   ├── unet_plus_plus.py
│   │   ├── deeplabv3.py
│   │   └── factory.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── callbacks.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── evaluate.py
│   │   └── visualization.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── seed.py
│       └── utils.py
│
├── models/
│   ├── checkpoints/
│   ├── best/
│   └── exported/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   │
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── health.py
│   │   │   │   ├── prediction.py
│   │   │   │   └── segmentation.py
│   │   │   └── dependencies.py
│   │   │
│   │   ├── services/
│   │   │   ├── inference.py
│   │   │   ├── preprocessing.py
│   │   │   └── postprocessing.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── prediction.py
│   │   │   └── response.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   │
│   │   └── utils/
│   │       └── image.py
│   │
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_prediction.py
│   │   └── test_segmentation.py
│   │
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── public/
│   │
│   ├── src/
│   │   ├── components/
│   │   │   ├── ImageUploader.jsx
│   │   │   ├── ImageViewer.jsx
│   │   │   ├── MaskViewer.jsx
│   │   │   └── Loading.jsx
│   │   │
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Segmentation.jsx
│   │   │   └── About.jsx
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   ├── hooks/
│   │   │   └── useSegmentation.js
│   │   │
│   │   ├── context/
│   │   │   └── AppContext.jsx
│   │   │
│   │   ├── assets/
│   │   ├── App.jsx
│   │   └── main.jsx
│   │
│   ├── package.json
│   └── Dockerfile
│
├── scripts/
│   ├── download_data.sh
│   ├── preprocess.sh
│   ├── train.sh
│   └── evaluate.sh
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── model.md
    └── api.md
```

# Pipeline

```
python -m ml.data.download => 
```