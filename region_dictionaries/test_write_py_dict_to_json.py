regions = {
    "Orbital": {
        "labels": [6],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "PrimarySomatosensory": {
        "labels": [16],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "SupplementalSomatosensory": {
        "labels": [18],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Auditory": {
        "labels": [20],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Retroplenial": {
        "labels": [24],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\', '/'),
        "volume_bar": 15,
        "volume_avgbar": 100
    },
    "PrimaryVisualArea": {
        "labels": [26],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Entorhinal": {
        "labels": [27],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Subiculum": {
        "labels": [28],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "CA1": {
        "labels": [31],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\', '/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "CA3": {
        "labels": [32],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\5xFAD\03subiculum.classifier".replace('\\', '/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "BLA": {
        "labels": [41],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\191209_BLA\BLAc_.classifier".replace('\\', '/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    "LGd": {
        "labels": [82],
        "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\', '/'),
        "volume_bar": 20,
        "volume_avgbar": 100
    },
    "Thalamus": {
        "labels": [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86],
        "classifier": r"K:/ProjectSpace/yt133/Labelmap/191209_BLA/BLAc_.classifier".replace('\\', '/'),
        "volume_bar": 10,
        "volume_avgbar": 100
    },
    #Something weird happens with Thalamus, a lot of java log, so Yuqi set up the last entry.
    # "delete": {
        # "labels": [83],
        # "classifier": r"K:\workstation\code\shared\img_processing\NeuronCounting\classifiers\200316auditory\short1.classifier".replace('\\','/'),
        # "volume_bar": 20,
        # "volume_avgbar": 100
    # }
}

import json
out_file="B:/ProjectSpace/hmm56/test_dict.json"
with open(out_file, "w") as f:
    json.dump(regions, f, indent=4)
