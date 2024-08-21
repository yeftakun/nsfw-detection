# NSFW Detection

This project is a NSFW image detection using the InceptionV3 model. It classifies images into five categories: `sexy`, `hentai`, `porn`, `neutral`, and `drawings`. The model is trained on a custom dataset and can be used to predict the category of any given image.

Dataset: deepghs/nsfw_detect (huggingface)

Model: [yeftakun/nsfw-detecion](https://huggingface.co/yeftakun/nsfw-detection/blob/main/inceptionv3_nsfw_model.h5) - `700 image` `5 epoch`



## Project Structure

```
├── img/
│ └── your-image.jpg # Example image for testing
├── nsfw_model_v1/
│ ├── sexy/
│ ├── hentai/
│ ├── porn/
│ ├── neutral/
│ └── drawings/
├── main.py # Script to load the model and classify images
├── training.py # Script to train the model
└── inceptionv3_nsfw_model.h5 # Trained model file
```


## Prerequisites

Ensure you have the following libraries installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy

You can install the required packages using pip:

```bash
pip install tensorflow keras numpy
```

## Training Model

Training the Model
To train the model on your custom dataset, ensure your dataset is organized in the following structure:

```
nsfw_model_v1/
├── sexy/
├── hentai/
├── porn/
├── neutral/
└── drawings/
```

Each subfolder should contain the respective images for each class.

Run the training.py script to train the model:

```
python training.py
```

The script will load the InceptionV3 model pre-trained on ImageNet, add custom layers on top, and fine-tune the model using your dataset. The trained model will be saved as inceptionv3_nsfw_model.h5.

## Classifying Images
To classify an image, place the image in the img/ folder and run the main.py script:

```
python main.py
```

Example output:

```
{
  "sexy": 0.01,
  "hentai": 0.00,
  "porn": 0.99,
  "neutral": 0.00,
  "drawings": 0.00
}
```