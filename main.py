from init import *  # import all the packages and modules we need
from src.utils import *


def main():
    args = cli.cli()
    model_pre, model_main, inner_model_2, inner_reduce_model_0, inner_reduce_model_1, inner_model_0, inner_model_1 = load_model()
    tags = load_class_list(model_main)
    y, sr, frames_reshaped = mel_spec_with_librosa(args.audio_source)
    predicted_labels, target_tag_prob, highest_prob, pred_class_num, pred_class_name = predict(model_pre, model_main, tags, frames_reshaped)
    grids, grids1 = gradcam_result(model_pre_output, model, class_num = 53)
    tensorboard_vis()
    plot_grad_cam(grids, grids1, args.output)


if __name__ == '__main__':
    main()
