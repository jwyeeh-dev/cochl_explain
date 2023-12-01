import os
import argparse
from init import * 
from src.utils import *
from src.cli import cli
from src.gradcam import *



def main():
    # Step 1: Load the model
    args = cli()
    model_pre, model_main, inner_model_2, inner_reduce_model_0, inner_reduce_model_1, inner_model_0, inner_model_1 = ModelUtils.load_model(args)

    # Step 2: Check the model type
    model_type = ModelUtils.check_model_type(args)
    print(f"Model Type: {model_type}")

    # Step 3: Load the class list
    tags = ModelUtils.load_class_list(args.main_model, model_main, args)

    # Step 4: Predict
    _, _, frames_reshaped = AudioUtils.mel_spec_with_librosa(args)
    predicted_labels, target_tag_prob, highest_prob, highest_class_num, highest_class_name = ModelUtils.predict(model_pre, model_main, tags, frames_reshaped)

    # Step 5: Generate GradCAM using TfExplain
    class_index = ModelUtils.find_class_num(args.class_name, tags)
    if model_type == 'ensemble':
        grad_cam_results1, grad_cam_results2 = GradCAMUtils.generate_grad_cam_tf_explain([inner_model_0, inner_model_1], frames_reshaped, class_index, ['m_main_0_Conv1', 'm_main_1_Conv1'], args)
    else:
        grad_cam_results1, grad_cam_results2 = GradCAMUtils.generate_grad_cam_tf_explain(model_main, frames_reshaped, class_index, 'conv_46', args)

    # Step 6: Visualization
    VisualizationUtils.plot_grad_cam(frames_reshaped, grad_cam_results1, grad_cam_results2, predicted_labels)

if __name__ == "__main__":
    main()