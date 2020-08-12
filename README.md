# EBAnO - Express
EBAnO (Explaining BlAck-box mOdels) is a simple and reliable tool that explain the prediction process of Deep Convolutional Neural Network.

EBAnO provides 2 main explainer APIs to produce:
1. Prediction-local explanations
2. Class-based model explanations

Further details about the project can be found at [EBAnO project page](https://ebano-ecosystem.github.io).

## Quick-Start
To have a quick look to EBAnO-Express and how to use it you can check 
1. The [EBAnO_express_Keras_VGG16.ipynb](https://github.com/EBAnO-Ecosystem/EBAnO-Express/blob/master/EBAnO_express_Keras_VGG16.ipynb) with a complete DEMO on VGG16.
2. The [EBAnO_express_other_models.ipynb](https://github.com/EBAnO-Ecosystem/EBAnO-Express/blob/master/EBAnO_express_other_models.ipynb) with a DEMO on other models.


## Documentation
### LocalExplanationModel
```
class ebano_express.explainer.LocalExplanationModel(input_image, class_of_interest, model, preprocess_func, max_features=10, k_pca=30, layers_to_analyze=None)
```
#### Parameters:
- **input_image: PIL.Image**  
    Target input image.
- **class_of_interest: int**  
    Target class of interest as index of model's output layer.
- **model: tensorflow.keras.models.Model**  
    Target deep convolutional model. 
    Even `tensorflow.python.keras.engine.training.Model` is accepted. 
- **preprocess_func: callable**  
    Defines the function used to preprocess the input image for the prediction process. 
    It should have one input parameter as PIL.Image and return a `numpy.ndarray`of shape (1, image_width, image_height, image_colors).
- **max_features: int, default=10**  
    Specifies the number of interpretable features to extract.
- **k_pca: int, default=30**  
    Specifies the number of components to reduce the DCNN hypercolumns dimensionality.
- **layers_to_analyze: list(int), default=None**  
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
One should produce local explanations only through a `LocalExplanationModel`.

```
class ebano_express.explainer.LocalExplanation(input_image, class_of_interest, features_map, model, preprocess_func)
```

#### Parameters:
- **input_image: PIL.Image**  
    Target input image.
- **class_of_interest: int**  
    Target class of interest as index of model's output layer.
- **features_map: numpy.array()**  
    A matrix of the same size of the input image that defines the feature id for each pixel.
- **model: tensorflow.keras.models.Model**  
    Target deep convolutional model. 
    Even `tensorflow.python.keras.engine.training.Model` is accepted.
- **preprocess_func: callable**  
    Defines the function used to preprocess the input image for the prediction process. 
    It should have one input parameter as PIL.Image and return a `numpy.ndarray`of shape (1, image_width, image_height, image_colors).

#### Methods:
- **fit_explanation(self, verbose=False)**  
    Fit the local explanation for the input_image.
- **show_features_map(self)**
    Make the plot of the features_map using matplotlib / pylab.  Return ```matplotlib.axes.Axes```.
- **show_visual_explanation(self)**
    Make the plot of the visual_explanation using matplotlib / pylab.  Return ```matplotlib.axes.Axes```.
- **show_numerical_explanation(self)**
    Make the plot of the numerica_explanation using matplotlib / pylab.  Return ```matplotlib.axes.Axes```.
- **get_interpretable_feature(self, f_idx)**
    Get the PIL.Image of the interpretable feature with id equal to f_idx.
- **get_numerical_explanation(self)**
    Get the raw numerical explanation as pandas.DataFrame.
- **get_visual_explanation(self)**
    Get the raw visual explanation as PIL.Image.

### ClassGlobalExplanationModel
The ```ClassGlobalExplanationModel``` exploits local explanations of input images to provide a detailed overview on the model behavior for a target class-of-interest.

```
class ebano_express.explainer.ClassGlobalExplanationModel(input_images, class_of_interest, model, preprocess_func, max_features=10, k_pca=30, layers_to_analyze=None)
```

#### Parameters:
- **input_images: list(PIL.Image)**  
    Target input image.
- **class_of_interest: int**  
    Target class of interest as index of model's output layer.
- **model: tensorflow.keras.models.Model**  
    Target deep convolutional model. 
    Even `tensorflow.python.keras.engine.training.Model` is accepted. 
- **preprocess_func: callable**  
    Defines the function used to preprocess the input image for the prediction process. 
    It should have one input parameter as PIL.Image and return a `numpy.ndarray`of shape (1, image_width, image_height, image_colors).
- **max_features: int, default=10**  
    Specifies the number of interpretable features to extract.
- **k_pca: int, default=30**  
    Specifies the number of components to reduce the DCNN hypercolumns dimensionality.
- **layers_to_analyze: list(int), default=None**  
    Defines the indexes of the convolutional layers of the model to analyze. Only convolutional layer indexes can be specified.
    If None the convolutional layers analyzed are the last <img src="https://render.githubusercontent.com/render/math?math=Log_2(n\_conv\_layers)">.

#### Attributes:
- **local_explanation_models: dict**
    It is a dictionary that contains the local explanation models computed for each input image.
- **global_explanation: list(dict)**
    The global explanation. It is composed by a list of dictionaries. 
    Each dictionary represents an image's feature recording the id of its image, its feature id, the PIL.Image of the feature, and a dictionary that defines its nPIR and nPIRP scores.
    ```python
    { 
      "image_id": int, 
      "feature_id": int, 
      "feature_img": PIL.Image, 
      "feature_scores": dict
    }
    ```
#### Methods:
- **fit_explanation(self, verbose=False)**  
    Fit the global explanation for the input_images.
- **show_global_explanation(self)**
    Make the plot of the global explanation using matplotlib / pylab.  Return ```matplotlib.axes.Axes```.

