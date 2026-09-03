"""Config for the LaTeX result tables built in results.ipynb.

Centralises the per-experiment dataset/method labels and paper-reported
baseline values that were previously hardcoded inline in each table cell.
"""

# Metric column sets used across tables
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'mean_iou', 'fom']
SANET_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'fom']

# Fixed NDWI thresholds per dataset (used by table cells and figure cells)
THRESHOLDS = {
    "LICS":   -0.11,
    "SWED":   -0.01,
    "SANet":   0.34,
    "TCUNet": -0.01,
}

# ---- Main results table ("Tables for thesis") ------------------------------
PAPER_ROWS = {
    'SWED': {
        'method':    r'U-Net~\cite{seale2022swed}',
        'accuracy':  0.937, 'precision': 0.916, 'recall': 0.948, 'f1': 0.922,
        'pos_iou':   0.875, 'mean_iou': None, 'fom': None,
    },
    'SANet': {
        'method':    r'SANet~\cite{cui2020sanet}',
        'accuracy':  0.986, 'precision': 0.984, 'recall': 0.987, 'f1': 0.986,
        'pos_iou':   None, 'mean_iou': None, 'fom': None,
    },
    'TCUNet': {
        'method':    r'TCUNet~\cite{Xiong2023}',
        'accuracy':  0.975, 'precision': None, 'recall': None, 'f1': 0.966,
        'pos_iou':   None, 'mean_iou': 0.935, 'fom': None,
    },
}

DS_TO_SAT = {
    'LICS': 'landsat', 'SWED': 'sentinel',
    'SANet': 'gaofen1', 'TCUNet': 'gaofen6',
}

DATASETS_ORDER = [
    ('LICS',   r'LICS \\ (Landsat)'),
    ('SWED',   r'SWED \\ (Sentinel-2)'),
    ('SANet',  r'SANet \\ (Gaofen-1)'),
    ('TCUNet', r'TCUNet \\ (Gaofen-6)'),
]

METHODS = [
    ('NDWI_fixed',  None),
    ('unet',        r'U-Net'),
    ('att_unet',    r'Attention U-Net'),
    ('r2_unet',     r'R2 U-Net'),
    ('r2att_unet',  r'R2 Attention U-Net'),
]

# ---- SANet detailed comparison table ----------------------------------------
PAPER_SANET_NDWI  = {'accuracy': 0.801, 'precision': 0.798, 'recall': 0.773, 'f1': 0.785, 'fom': None}
PAPER_SANET_UNET  = {'accuracy': 0.947, 'precision': 0.9200, 'recall': 0.972, 'f1': 0.945, 'fom': None}
PAPER_SANET_SANET = {'accuracy': 0.986, 'precision': 0.984, 'recall': 0.987, 'f1': 0.986, 'fom': None}

# ---- Augmentation table (Experiment 3) --------------------------------------
AUG_MAP = {
    'none':           'None',
    'geometric':      'Geometric',
    'gaussian_noise': 'Gaussian Noise',
    'salt_pepper':    r'Salt \& Pepper',
    'contrast':       'Contrast',
    'combined':       'Combined',
}
AUG_ORDER = ['none', 'geometric', 'gaussian_noise', 'salt_pepper', 'contrast', 'combined']
AUG_DATASETS = [
    ('landsat',  r'LICS \\ (Landsat)'),
    ('sentinel', r'SWED \\ (Sentinel-2)'),
]

# ---- Fine-tuning table (Experiment 4) ----------------------------------------
FT_DATASETS = [('LICS', 'landsat'), ('SWED', 'sentinel')]

# Fine-tuning rows per dataset: (pretrained, encoder, freeze, display_label)
FT_ROWS = {
    'LICS': [
        ('imagenet',    'resnet50',  False, 'ImageNet (ResNet-50)'),
        ('imagenet',    'resnet50',  True,  'ImageNet (ResNet-50)'),
        ('bigearthnet', 'resnet101', False, 'BigEarthNet (ResNet-101)'),
        ('bigearthnet', 'resnet101', True,  'BigEarthNet (ResNet-101)'),
    ],
    'SWED': [
        ('imagenet',    'resnet101', False, 'ImageNet (ResNet-101)'),
        ('imagenet',    'resnet101', True,  'ImageNet (ResNet-101)'),
        ('bigearthnet', 'resnet18',  False, 'BigEarthNet (ResNet-18)'),
        ('bigearthnet', 'resnet18',  True,  'BigEarthNet (ResNet-18)'),
    ],
}

# ---- Reduced-band table (Experiment 5) ----------------------------------------
DATASETS_EXP5 = ['LICS', 'SWED', 'SANet', 'TCUNet']
