import re
import os
import numpy as np
import cv2
import librosa
import pandas as pd
from matplotlib import pyplot as plt
from tf_explain.core.grad_cam import GradCAM as TfExplainGradCAM
from tf_keras_vis.gradcam import Gradcam as TfKerasVisGradCAM
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from init import *
from src.utils import *

class GradCAMUtils:
    def __init__(self):
        pass

    @staticmethod
    def _preprocess_input(input):
        """
        Preprocesses the input image before generating GradCAM.

        Parameters:
        - input (str): Path to the input source.

        Returns:
        - img (numpy.ndarray): Preprocessed image.
        """
        data = np.expand_dims(input, axis=0)
        
        return data

    @classmethod
    def generate_grad_cam_tf_explain(cls, model, model_1, model_pre_output, class_num, layer_name, args):
        """
        Generates GradCAM using TfExplain.

        Parameters:
        - model: The Keras model.
        - model_pre_output (str): Output from the pre-processing model.
        - class_index (int): Index of the target class.
        - layer_name (tuple or str): Name of the target layer.

        Returns:
        - None (Displays the generated GradCAM).
        """
        data = []
        grids1 = []
        grids2 = []

    
        # Apply GradCAM
        explainer = TfExplainGradCAM()

        for i in range(model_pre_output.shape[0]):
            data = cls._preprocess_input(model_pre_output[i])
            data = (data, None)
            print(data[0].shape)

            if ModelUtils.check_model_type(args) == 'single':
                cam1 = explainer.explain(data, model, class_index=class_num, layer_name=layer_name)
                cam2 = explainer.explain(data, model, class_index=class_num, layer_name=layer_name)

                grids1.append(cam1)
                grids2.append(cam2)

            elif ModelUtils.check_model_type(args) == 'ensemble':
                cam1 = explainer.explain(data, model, class_index=class_num, layer_name=layer_name[0])
                cam2 = explainer.explain(data, model_1, class_index=class_num, layer_name=layer_name[1])
                grids1.append(cam1)
                grids2.append(cam2)
            else:
                ("Error: Model Type is not defined")

        return grids1, grids2

    @staticmethod
    def generate_grad_cam_tf_keras_vis(model, model_pre_output, class_index, layer_name, args):
        """
        Applies GradCAM and generates visualization.

        Parameters:
        - model_pre_output: Output from the pre-processing model.
        - model: The Keras model.
        - class_num (int): Index of the target class.

        Returns:
        - Tuple: GradCAM results for different branches.
        """
        grids1 = []
        grids2 = []

        # Apply GradCAM
        for i in range(model_pre_output.shape[0]):
            model_data = GradCAMUtils._preprocess_input(model_pre_output[i])

            # Create GradCAM
            if ModelUtils.check_model_type(args) == 'single':
                gradcam1 = TfKerasVisGradCAM(model, model_modifier=ReplaceToLinear(), clone=True)
                gradcam2 = TfKerasVisGradCAM(model, model_modifier=ReplaceToLinear(), clone=True)
                
                # Generate heatmap with GradCAM++
                cam1 = gradcam1(CategoricalScore(class_index), model_data)
                cam2 = gradcam2(CategoricalScore(class_index), model_data)

                cam1 = cam1.reshape(128,96,1)
                cam2 = cam2.reshape(128,96,1)

                # Append to results
                grids1.append(cam1)
                grids2.append(cam2)

            elif ModelUtils.check_model_type(args) == 'ensemble':
                gradcam1 = TfKerasVisGradCAM(model[0], model_modifier=ReplaceToLinear(), clone=True)
                gradcam2 = TfKerasVisGradCAM(model[1], model_modifier=ReplaceToLinear(), clone=True)
                
                # Generate heatmap with GradCAM++
                cam1 = gradcam1(CategoricalScore(class_index), model_data)
                cam2 = gradcam2(CategoricalScore(class_index), model_data)

                cam1 = cam1.reshape(128,96,1)
                cam2 = cam2.reshape(128,96,1)

                # Append to results
                grids1.append(cam1)
                grids2.append(cam2)

            

        return grids1, grids2