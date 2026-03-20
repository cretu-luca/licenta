TASKS = {
    'Alternating': {
        'col': 'Alternating',
        'num_classes': 2,
        'min_val': 0,
        'type': 'binary',
    },
    'Crossing Number': {
        'col': 'Crossing Number',
        'num_classes': 11,
        'min_val': 3,
    },
    'Unknotting Number': {
        'col': 'Unknotting Number',
        'num_classes': 5,
        'min_val': 1,
    },
    'Signature': {
        'col': 'Signature',
        'num_classes': 19,
        'min_val': -10,
    },
    'Determinant': {
        'col': 'Determinant',
        'num_classes': 663,
        'min_val': 1,
    },
    'Genus-3D': {
        'col': 'Genus-3D',
        'num_classes': 5,
        'min_val': 1,
    },
    'Genus-4D': {
        'col': 'Genus-4D',
        'num_classes': 6,
        'min_val': 0,
    },
    'Genus-4D (Top.)': {
        'col': 'Genus-4D (Top.)',
        'num_classes': 6,
        'min_val': 0,
    },
    'Rasmussen s': {
        'col': 'Rasmussen <i>s</i>',
        'num_classes': 19,
        'min_val': -8,
    },
    'Ozsvath-Szabo tau': {
        'col': 'Ozsvath-Szabo <i>tau</i>',
        'num_classes': 10,
        'min_val': -4,
    },
    'Arf': {
        'col': 'Arf Invariant',
        'num_classes': 2,
        'min_val': 0,
    },
}

TRAINING = {
    'hidden_dim': 128,
    'num_layers': 4,
    'lr': 0.0005,
    'batch_size': 32,
    'num_epochs': 400,
    'patience': 100,
    'split': (0.8, 0.1, 0.1),
    'seed': 42,
}
