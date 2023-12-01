import numpy as np
import cv2
from matplotlib import pyplot as plt
from tf_explain.core.grad_cam import GradCAM as TfExplainGradCAM
from tf_keras_vis.gradcam import Gradcam as TfKerasVisGradCAM
import tensorflow as tf

class GradCAM:
    def __init__(self, model):
        self.model = model

    def _preprocess_input(self, image_path):
        # 이미지 전처리 로직을 여기에 추가
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # 모델 입력 크기에 맞게 조절
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # 정규화
        return img

    def generate_grad_cam_tf_explain(self, image_path, class_index, layer_name):
        img = self._preprocess_input(image_path)

        explainer = TfExplainGradCAM()
        grid = explainer.explain(
            validation_data=(img, None),
            model=self.model,
            layer_name=layer_name,
            class_index=class_index,
        )

        # 시각화
        plt.imshow(grid)
        plt.show()

    def generate_grad_cam_tf_keras_vis(self, image_path, class_index, layer_name):
        img = self._preprocess_input(image_path)

        explainer = TfKerasVisGradCAM(self.model, layer_name)
        grid = explainer(img, class_index)

        # 시각화
        plt.imshow(grid)
        plt.show()


        