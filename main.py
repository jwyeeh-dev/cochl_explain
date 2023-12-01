import os
import argparse
from init import *  # import all the packages and modules we need
from src.utils import *



def main():
    # Step 1: Load the model
    args = cli.cli()
    model_pre, model_main, inner_model_2, inner_reduce_model_0, inner_reduce_model_1, inner_model_0, inner_model_1 = ModelUtils.load_model(args)

    # Step 2: Check the model type
    model_type = ModelUtils.check_model_type(args)
    print(f"Model Type: {model_type}")

    # Step 3: Load the class list
    tags = ModelUtils.load_class_list(args.model_path, model_main, args)

    # Step 4: Predict
    y, sr, frames_reshaped = AudioUtils.mel_spec_with_librosa(args)
    predicted_labels, target_tag_prob, highest_prob, pred_class_num, pred_class_name = ModelUtils.predict(model_pre, model_main, tags, frames_reshaped)

    # Step 5: Generate GradCAM using TfExplain
    class_index = ModelUtils.find_class_num(args.class_name, tags)
    if model_type == 'ensemble':
        grad_cam_results1, grad_cam_results2 = GradCAMUtils.generate_grad_cam_tf_explain([inner_reduce_model_0, inner_reduce_model_1], frames_reshaped, class_index, 'activation_2')
    else:
        grad_cam_results1, grad_cam_results2 = GradCAMUtils.generate_grad_cam_tf_explain([model_main], frames_reshaped, class_index, 'activation')

    # Step 6: Visualization
    VisualizationUtils.plot_grad_cam(frames_reshaped.squeeze(), grad_cam_results1, grad_cam_results2, predicted_labels)

if __name__ == "__main__":
    main()