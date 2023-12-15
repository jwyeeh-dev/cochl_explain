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
        if re.search(r'ensemble', args.main_model, re.IGNORECASE):
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


        if ModelUtils.check_model_type(args) == 'ensemble': 
            inner_model_2 = model_main.get_layer('model_2')
            inner_reduce_model_0 = model_main.get_layer('model_2').get_layer('reduced_model_0')
            inner_reduce_model_1 = model_main.get_layer('model_2').get_layer('reduced_model_1')
            inner_model_0 = model_main.get_layer('model_2').get_layer('reduced_model_0').get_layer('model')
            inner_model_1 = model_main.get_layer('model_2').get_layer('reduced_model_1').get_layer('model_1')
        
        elif ModelUtils.check_model_type(args) == 'single':
            inner_model_2 = None
            inner_reduce_model_0 = None
            inner_reduce_model_1 = None
            inner_model_0 = None
            inner_model_1 = None
        
        else: 
            print("No model found.")
            print("Please write your own model before you start.")

        return model_pre, model_main, inner_model_2, inner_reduce_model_0, inner_reduce_model_1, inner_model_0, inner_model_1

    @staticmethod
    def load_tflite_model():
        """
        Loads and returns TFLite models.

        Returns:
        - Tuple of TFLite models and inner models.
        """

        interpreter_pre = tf.lite.Interpreter(model_path=args.pre_model)
        interpreter_pre.allocate_tensors()

        input_details_pre = interpreter_pre.get_input_details()
        output_details_pre = interpreter_pre.get_output_details()



        return interpreter, input_details, output_details


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
                tags = pd.read_csv('./assets/class_list/cochldb.tags.csv')

            else:
                all_layers = [layer.name for layer in model.layers]
                activation_layers = [layer for layer in all_layers if re.search(r'activation', layer, re.IGNORECASE)]

                last_activation_layer_name = activation_layers[-1]
                last_activation_layer = model.get_layer(last_activation_layer_name)

                class_cnt = last_activation_layer.output_shape[1]

                # According to the number of classes, load the class list
                if class_cnt == 752: tags = pd.read_csv('./assets/class_list/cochldb.tags.orig.csv')
                elif class_cnt == 772: tags = pd.read_csv('./assets/class_list/cochldb.tags.orig.772.csv')
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
    def find_last_conv_layer(model_main):
        """
        Finds the last convolutional layer.

        Parameters:
        - model_main: The Keras model.

        Returns:
        - str: Name of the last convolutional layer.
        """

        last_conv_layer = None
        for layer in reversed(model_main.layers):
            if len(layer.output_shape) == 4:
                last_conv_layer = layer.name
                break

        return last_conv_layer

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

        args = cli()

        # Model input
        target_tag_prob = []
        highest_prob = []
        highest_class_num = []
        highest_class_name = []

        class_num = ModelUtils.find_class_num(args.class_name, tags) 

        # 0th: model_pre
        model_pre_output = model_pre.predict(frames_reshaped)

        # 1st: model_main
        predicted_labels = model_main.predict(model_pre_output)

        # Target tag probability (want to be predicted)
        for i in range(len(model_pre_output)):
            target_tag_prob.append(predicted_labels[i,class_num])

            # Highest probability tag (result)
            highest_prob.append(predicted_labels[i].max())
            highest_class_num.append(predicted_labels[i].argmax())
            highest_class_name.append(tags['tags'][highest_class_num[i]])

        # ------------------ TARGET RESULT ------------------
        print("\n")
        print("="*25 + "TARGET RESULT" + "="*25)
        print(f"Probability         | {target_tag_prob}")
        print(f"Target_class_number | {class_num}")
        print(f"Target_class        | {args.class_name}")
        print("="* 60)

        # ------------------ PREDICTED RESULT ------------------
        print("\n")
        print("="*20 + "MOST PREDICTED RESULT" + "="*20)
        print(f"Probability            | {highest_prob}")
        print(f"Predicted_class_number | {highest_class_num}")
        print(f"Predicted_class        | {tags['tags'][highest_class_num].values}")
        print("="* 60)

        return model_pre_output, predicted_labels, target_tag_prob, highest_prob, highest_class_num, highest_class_name


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
    def plot_grad_cam(model_pre_output, grids1, grids2, class_name, target_tag_prob, seconds):
        plt.figure(figsize=(20, 5))
        plt.imshow(model_pre_output[seconds], aspect='auto', origin='lower')
        plt.imshow(grids1[seconds][:,:], alpha=0.4, aspect='auto', origin='lower', cmap='jet')
        plt.imshow(grids2[seconds][:,:], alpha=0.4, aspect='auto', origin='lower', cmap='jet')
        plt.colorbar(format='%+02.0f')
        plt.title(f'Probability : {target_tag_prob[seconds]:.4f}', fontsize=30)
        plt.ylabel('Mel Frequency', fontsize=30)
        plt.xlabel(f'Time Frame : 1sec', fontsize=30)
        plt.savefig('./assets/_output/gradcam.png')
        plt.close()

    def subplot_grad_cam(model_pre_output, grids1, grids2, predicted_labels):
        
        """
        Plots GradCAM visualizations.

        Parameters:
        - model_pre_output: Output from the pre-processing model.
        - grids: GradCAM results for the first model.
        - grids1: GradCAM results for the second model.
        """
        mid = model_pre_output.shape[0] // 2
        fig, axs = plt.subplots(2, mid, figsize=(50, 35), facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
        print(model_pre_output.shape[0])
        for i in range(model_pre_output.shape[0]):
            
            if i < mid:  
                axs[0][i].imshow(model_pre_output[i], aspect='auto', origin='lower')
                axs[0][i].imshow(grids1[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')  
                axs[0][i].imshow(grids2[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')
                axs[0][i].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)
                axs[0][i].set_ylabel('Mel Frequency', fontsize=35)
                axs[0][i].set_xlabel(f'Time Frame : {i+1}sec', fontsize=35)

            elif i >= mid:
                axs[1][i-mid].imshow(model_pre_output[i], aspect='auto', origin='lower')
                axs[1][i-mid].imshow(grids1[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')  
                axs[1][i-mid].imshow(grids2[i], alpha=0.3, aspect='auto', origin='lower', cmap='jet')
                axs[1][i-mid].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)
                axs[1][i-mid].set_ylabel('Mel Frequency', fontsize=35)
                axs[1][i-mid].set_xlabel(f'Time Frame: {i+1}sec', fontsize=35)
        
        fig.savefig('./assets/_output/gradcam.png', axs)
        plt.close()

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
            mean_activations_red = grids[i,:, :, 0].mean(axis=mode)

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

        plt.savefig('./assets/_output/activation_hist.png', axs)
        plt.close()

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
            mean_activations_red = grids[i,:, :].mean(axis=mode)

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

        plt.savefig('./assets/_output/activation_hist_grad.png', axs)
        plt.close()
 

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

        y, sr = librosa.load(args.audio_source, sr=22050)  # 로드 시 샘플링 레이트를 22050으로 설정
        frames = librosa.util.frame(y, frame_length=22050, hop_length=22050).T # 1초 단위로 나눔
        frames_reshaped = frames.reshape(frames.shape[0], 1, 22050) # 모델 입력을 위해 적절한 형태로 변환

        return y, sr, frames_reshaped
