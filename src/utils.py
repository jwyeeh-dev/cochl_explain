import re
import os
import numpy as np
import cv2
import librosa
import pandas as pd
from matplotlib import pyplot as plt
from tf_explain.core.grad_cam import GradCAM as TfExplainGradCAM
from tf_keras_vis.gradcam import Gradcam as TfKerasVisGradCAM
from tf_keras_vis.utils import CategoricalScore, ReplaceToLinear
from init import *

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
        data = (data, None)
        return data

    @classmethod
    def generate_grad_cam_tf_explain(cls, model, model_pre_output, class_num, layer_name):
        """
        Generates GradCAM using TfExplain.

        Parameters:
        - model: The Keras model.
        - model_pre_output (str): Output from the pre-processing model.
        - class_index (int): Index of the target class.
        - layer_name (str): Name of the target layer.

        Returns:
        - None (Displays the generated GradCAM).
        """
        explainer = TfExplainGradCAM()

        data = []
        grids1 = []
        grids2 = []


        for i in range(model_pre_output.shape[0]):
            data = cls._preprocess_input(model_pre_output[i])
            if model.shape == 1:
                cam1 = explainer.explain(data, model, class_index=class_num, layer_name=layer_name)
                cam2 = explainer.explain(data, model, class_index=class_num, layer_name=layer_name)
            elif model.shape > 1:
                cam1 = explainer.explain(data, model[0], class_index=class_num, layer_name=layer_name)
                cam2 = explainer.explain(data, model[1], class_index=class_num, layer_name=layer_name)
            
            grids1.append(cam1)
            grids2.append(cam2)

        return grids1, grids2

    @staticmethod
    def generate_grad_cam_tf_keras_vis(cls, model_pre_output, model, class_index):
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
            model_data = cls._preprocess_input(model_pre_output[i])

            # Create GradCAM
            if model.shape == 1:
                gradcam1 = TfExplainGradCAM(model, model_modifier=ReplaceToLinear(), clone=True)
                gradcam2 = TfExplainGradCAM(model, model_modifier=ReplaceToLinear(), clone=True)
            elif model.shape > 1:
                gradcam1 = TfExplainGradCAM(model[0], model_modifier=ReplaceToLinear(), clone=True)
                gradcam2 = TfExplainGradCAM(model[1], model_modifier=ReplaceToLinear(), clone=True)

            # Generate heatmap with GradCAM++
            data = (model_data, None)
            cam1 = gradcam1(CategoricalScore(class_index), model_data)
            cam2 = gradcam2(CategoricalScore(class_index), model_data)

            cam1 = cam1.reshape(128,96,1)
            cam2 = cam2.reshape(128,96,1)

            # Append to results
            grids1.append(cam1)
            grids2.append(cam2)

        return grids1, grids2


class ModelUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def check_model_type(args):
        """
        Checks the type of the model.

        Parameters:
        - model_path (str): Path to the model.

        Returns:
        - int: Number of models.
        """
        if re.search(r'ensemble', args.model_path, re.IGNORECASE):
            return 'ensemble'
        else:
            return 'single'

    @staticmethod
    def load_model(args):
        """
        Loads and returns Keras models.

        Returns:
        - Tuple of Keras models and inner models.
        """
        model_pre = keras.models.load_model(args.pre_model, compile=False)
        model_main = keras.models.load_model(args.main_model, compile=False)

        pattern = re.compile(r'\bensemble\b')
        match = re.search(pattern, args.main_model)

        if match:
            inner_model_2 = model_main.get_layer('model_2')
            inner_reduce_model_0 = model_main.get_layer('model_2').get_layer('reduced_model_0')
            inner_reduce_model_1 = model_main.get_layer('model_2').get_layer('reduced_model_1')
            inner_model_0 = model_main.get_layer('model_2').get_layer('reduced_model_0').get_layer('model')
            inner_model_1 = model_main.get_layer('model_2').get_layer('reduced_model_1').get_layer('model_1')
        else:
            inner_model_2 = None
            inner_reduce_model_0 = None
            inner_reduce_model_1 = None
            inner_model_0 = None
            inner_model_1 = None

        return model_pre, model_main, inner_model_2, inner_reduce_model_0, inner_reduce_model_1, inner_model_0, inner_model_1

    @staticmethod
    def load_class_list(model_path, model, args):
        """
        Loads class list based on the model's class count.

        Parameters:
        - model_path : Path to the model (name)
        - model: The Keras model.
        - args : Command-line arguments.

        Returns:
        - DataFrame: DataFrame containing class information.
        """
        if args.class_list:
            tags = pd.read_csv(args.class_list)

        else:
            if re.search(r'ensemble', model_path, re.IGNORECASE):
                tags = pd.read_csv('./model_inferences/cochldb.tags.csv')

            else:
                all_layers = [layer.name for layer in model.layers]
                activation_layers = [layer for layer in all_layers if re.search(r'activation', layer, re.IGNORECASE)]

                last_activation_layer_name = activation_layers[-1]
                last_activation_layer = model.get_layer(last_activation_layer_name)

                class_cnt = last_activation_layer.output_shape[1]

                # According to the number of classes, load the class list
                if class_cnt == 752: tags = pd.read_csv('./model_inferences/cochldb.tags.orig.csv')
                elif class_cnt == 772: tags = pd.read_csv('./model_inferences/cochldb.tags.orig.772.csv')
                else: 
                    print("No class list found.")
                    print("Please write your own class list before you start.")    

        return tags

    @staticmethod
    def find_class_num(tag_name, tags_df):
        """
        Finds the class number based on the tag name.

        Parameters:
        - tag_name (str): The target class tag name.
        - tags_df (DataFrame): DataFrame containing class information.

        Returns:
        - int: Class number.
        """
        class_num = tags_df[tags_df['tags'] == tag_name].index[0]
        return class_num

    @staticmethod
    def predict(model_pre, model_main, tags, frames_reshaped):
        """
        Makes predictions using the given models.

        Parameters:
        - model_pre: Pre-processing model.
        - model_main: Main prediction model.
        - tags (DataFrame): DataFrame containing class information.
        - frames_reshaped: Reshaped frames for prediction.

        Returns:
        - Tuple: Predicted labels, target tag probability, highest probability, predicted class number, predicted class name.
        """
        args = cli.cli()

        # Model input
        most_predicted = []
        target_predicted = []
        model_outputs = []
        class_num = ModelUtils.find_class_num(args.class_name, tags) 

        # 0th: model_pre
        model_pre_output = model_pre.predict(frames_reshaped)

        # 1st: model_main
        # Target tag probability (want to be predicted)
        for i in range(len(model_pre_output)):
            predicted_labels = model_main.predict(model_pre_output[i])
            target_tag_prob = predicted_labels[i][class_num]

        # Highest probability tag (result)
            highest_prob = predicted_labels.max()
            pred_class_num = predicted_labels.argmax()
            pred_class_name = tags['tags'][pred_class_num] 

        # ------------------ TARGET RESULT ------------------
        print("="*20 + "RESULT" + "="*20)
        print(f"Probability         | {target_tag_prob}")
        print(f"Target_class_number | {class_num}")
        print(f"Target_class        | {args.class_name}")
        print("="* 46)

        # ------------------ PREDICTED RESULT ------------------
        print("="*20 + "RESULT" + "="*20)
        print(f"Probability            | {highest_prob}")
        print(f"Predicted_class_number | {pred_class_num}")
        print(f"Predicted_class        | {tags['tags'][pred_class_num]}")
        print("="* 46)

        return predicted_labels, target_tag_prob, highest_prob, pred_class_num, pred_class_name

class VisualizationUtils:
    def __init__(self):
        pass



    @staticmethod
    def tensorboard_vis():
        """
        Launches TensorBoard for visualization.
        """
        os.system('tensorboard --logdir ./logs/fit --host localhost --port 8088')

    @staticmethod
    def plot_grad_cam(model_pre_output, grids1, grids2, predicted_labels):
        """
        Plots GradCAM visualizations.

        Parameters:
        - model_pre_output: Output from the pre-processing model.
        - grids: GradCAM results for the first model.
        - grids1: GradCAM results for the second model.
        """
        fig, axs = plt.subplots(2, 5, figsize=(50, 35), facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

        for i in range(model_pre_output.shape[0]):
            if i < 5:  
                axs[0][i].imshow(model_pre_output[i], aspect='auto', origin='lower')
                axs[0].imshow(grids1[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')  
                axs[0].imshow(grids2[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')
                axs[0][i].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)
                axs[0][i].set_ylabel('Mel Frequency', fontsize=35)
                axs[0][i].set_xlabel(f'Time Frame : {i+1}sec', fontsize=35)

            elif i >= 5:
                axs[1][i-5].imshow(model_pre_output[i], aspect='auto', origin='lower')
                axs[1].imshow(grids1[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')  
                axs[1].imshow(grids2[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')
                axs[1][i-5].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)
                axs[1][i-5].set_ylabel('Mel Frequency', fontsize=35)
                axs[1][i-5].set_xlabel(f'Time Frame: {i+1}sec', fontsize=35)

        plt.show()

    @staticmethod
    def plot_activation_histograms(grids, predicted_labels, mode=0):
        """
        Plots activation histograms.

        Parameters:
        - grids: GradCAM results.
        - predicted_labels: Predicted labels.
        - mode (int): Mode for plotting.

        Mode:
        - 0: Vertical bar plot.
        - 1: Horizontal bar plot.
        """
        fig, axs = plt.subplots(2, 5, figsize=(50, 35), facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

        for i in range(len(grids)):
            # Calculate mean activations of the red channel
            mean_activations_red = grids[i][:, :, 0].mean(axis=mode)

            if mode == 0:
                axs[i // 5, i % 5].bar(range(len(mean_activations_red)), mean_activations_red, alpha=0.5, color='red')
                axs[i // 5, i % 5].set_yticks(range(len(mean_activations_red)))
                axs[i // 5, i % 5].set_yticklabels(range(1, len(mean_activations_red) + 1))
                axs[i // 5, i % 5].set_xlabel('Time Frame', fontsize=35)
            elif mode == 1:
                axs[i // 5, i % 5].barh(range(len(mean_activations_red)), mean_activations_red, alpha=0.5, color='red')
                axs[i // 5, i % 5].set_xlabel('Average Activation', fontsize=20)
                axs[i // 5, i % 5].set_yticks(range(len(mean_activations_red)))
                axs[i // 5, i % 5].set_xticklabels(range(1, len(mean_activations_red) + 1))
                axs[i // 5, i % 5].set_ylabel('Mel Frequency', fontsize=35)

            # Set axis and title
            axs[i // 5, i % 5].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)

        plt.show()

    @staticmethod
    def plot_activation_histograms_grad(grids, predicted_labels, mode=0):
        """
        Plots activation histograms for GradCAM results.

        Parameters:
        - grids: GradCAM results.
        - predicted_labels: Predicted labels.
        - mode (int): Mode for plotting.

        Mode:
        - 0: Vertical bar plot.
        - 1: Horizontal bar plot.
        """
        fig, axs = plt.subplots(2, 5, figsize=(50, 35), facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

        for i in range(len(grids)):
            # Red 채널의 평균 activation 값을 계산합니다.
            mean_activations_red = grids[i][:, :].mean(axis=mode)

            if mode == 0:
                axs[i // 5, i % 5].bar(range(len(mean_activations_red)), mean_activations_red, alpha=0.5, color='red')
                axs[i // 5, i % 5].set_ylabel('Average Activation', fontsize=20)
                axs[i // 5, i % 5].set_yticks(range(len(mean_activations_red)))
                axs[i // 5, i % 5].set_yticklabels(range(1, len(mean_activations_red) + 1))
                axs[i // 5, i % 5].set_xlabel('Time Frame', fontsize=35)
            elif mode == 1:
                axs[i // 5, i % 5].barh(range(len(mean_activations_red)), mean_activations_red, alpha=0.5, color='red')
                axs[i // 5, i % 5].set_xlabel('Average Activation', fontsize=20)
                axs[i // 5, i % 5].set_yticks(range(len(mean_activations_red)))
                axs[i // 5, i % 5].set_xticklabels(range(1, len(mean_activations_red) + 1))
                axs[i // 5, i % 5].set_ylabel('Mel Frequency', fontsize=35)

            axs[i // 5, i % 5].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)

        plt.show()


class AudioUtils:
    def __init__(self):
        pass

    @staticmethod
    def mel_spec_with_librosa(args):
        """
        Compute mel spectrogram using librosa.

        Parameters:
        - audio_source (str): Path to the audio source file.

        Returns:
        - y (numpy.ndarray): Audio time series.
        - sr (int): Sample rate of `y`.
        - frames_reshaped (numpy.ndarray): Reshaped mel spectrogram frames.
        """
        y, sr = librosa.load(args.audio_source, sr=22050)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        frames_reshaped = mel_spectrogram.reshape(1, 128, 87, 1)

        return y, sr, frames_reshaped
    
