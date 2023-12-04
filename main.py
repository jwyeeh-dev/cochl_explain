import os
import argparse
from init import * 
from src.utils import *
from src.cli import cli
from src.gradcam import *

def main():
    # Step 1: Load the model
    args = cli()
    model_utils = ModelUtils()
    
    model_pre, model_main, inner_model_2, inner_reduce_model_0, inner_reduce_model_1, inner_model_0, inner_model_1 = model_utils.load_model(args)


    # Step 2: Check the model type
    model_type = model_utils.check_model_type(args)
    print(f"Model Type: {model_type}")

    # Step 3: Load the class list
    tags = model_utils.load_class_list(args.main_model, model_main, args)

    # Step 4: Predict
    _, _, frames_reshaped = AudioUtils.mel_spec_with_librosa(args)
    model_pre_output, predicted_labels, target_tag_prob, highest_prob, highest_class_num, highest_class_name = model_utils.predict(model_pre, model_main, tags, frames_reshaped)

    # Step 5: Generate GradCAM using TfExplain
    class_index = model_utils.find_class_num(args.class_name, tags)
    gradcam_utils = GradCAMUtils()
    if model_type == 'ensemble':
        #grad_cam_results1, grad_cam_results2 = gradcam_utils.generate_grad_cam_tf_explain(inner_model_0, inner_model_1, model_pre_output, class_index, ['m_main_0_Conv_1', 'm_main_1_Conv_1'], args)
        grad_cam_results1, grad_cam_results2 = gradcam_utils.generate_grad_cam_tf_keras_vis([inner_model_0, inner_model_1], model_pre_output, class_index, ['m_main_0_Conv_1', 'm_main_1_Conv_1'], args)
    else:
        grad_cam_results1, grad_cam_results2 = gradcam_utils.generate_grad_cam_tf_keras_vis(model_main, model_pre_output, class_index, 'conv_46', args)
        
    # Step 6: Visualization
    VisualizationUtils.plot_grad_cam(model_pre_output, grad_cam_results1, grad_cam_results2, args.class_name, target_tag_prob=target_tag_prob, seconds=args.target_sec)


if __name__ == "__main__":
    main()