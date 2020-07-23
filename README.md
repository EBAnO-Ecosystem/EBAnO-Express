# EBAnO - Express
EBAnO (Explaining BlAck-box mOdels) is a simple and reliable tool that explain the prediction process of Deep Convolutional Neural Network.

EBAnO provides 2 main explainer APIs to produce:
1. Prediction-local explanations
2. Class-based model explanations

## Prediction-local explanation

### LocalExplanationModel
```
class ebano_express.explainer.LocalExplanationModel(input_image, class_of_interest, model, preprocess_func, max_features=10, k_pca=30, layers_to_analyze=None)
```
#### Parameters:
- **input_image: PIL.Image**  
    Target input image.
- **class_of_interest: int**  
    Target class of interest as index of model's output layer.
- **model:tensorflow.keras.models.Model**  
    Target deep convolutional model. 
    Even `tensorflow.python.keras.engine.training.Model` is accepted. 
- **preprocess_func:callable**  
    Defines the function used to preprocess the input image for the prediction process. 
    It should have one input parameter as PIL.Image and return a `numpy.ndarray`of shape (1, image_width, image_height, image_colors).
- **max_features:int, default=10**  
    Specifies the number of interpretable features to extract.
- **k_pca:int, default=30**  
    Specifies the number of components to reduce the DCNN hypercolumns dimensionality.
- **layers_to_analyze:list(int), default=None**  
    Defines the indexes of the convolutional layers of the model to analyze. Only convolutional layer indexes can be specified.
    If None the convolutional layers analyzed are the last <img src="https://render.githubusercontent.com/render/math?math=Log_2(n\_conv\_layers)">.


#### Attributes:
- **best_explanation:ebano_express.explainer.LocalExplanation**  
    Contains the most informative explanation computed for the input image.
- **local_explanations: dict**  
    A dictionary containing all the produced explanations for the input image.
    Dictionary structure:  
    ```python
        {
         2: ebano_express.explainer.LocalExplanation, 
         3: ebano_express.explainer.LocalExplanation,
         ...,
         max_features: ebano_express.explainer.LocalExplanation
        }
    ```
#### Methods:
- **fit_explanation(self, verbose=False)**  
    Fit the local explanations for the given `LocalExplanationModel` configuration.

### LocalExplanation
A `LocalExplanationModel` produces local explanations represented by the `LocalExplanation` class.
One should build local explanations exploiting a `LocalExplanationModel`.

**TODO completare**
#### Attributes:
#### Methods:

## Class-global explanations
### ClassGlobalExplanationModel
**TODO completare**
#### Parameters:
#### Attributes:
#### Methods:

### ClassGlobalExplanation
**TODO completare**
#### Attributes:
#### Methods:

## Quick-Start
### Import EBAnO Express

```python
from ebano_express.explainer import LocalExplanationModel
```

### Produce a prediction-local explanation
1. Load a Keras convolutional model
    ```python
   from tensorflow.keras.applications import VGG16
   VGG16_model = VGG16(include_top=True, weights='imagenet', classes=1000) 
   ```
2. Load target image as PILLOW Image
    ```python
   from tensorflow.python.keras.preprocessing import image
   
   image_url = "path_to_input_image"
   target_size = (224, 224)
   image.load_img(image_url, target_size=target_size)
   ```

**TODO completare**