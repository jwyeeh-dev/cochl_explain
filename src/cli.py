import argparse

def cli():
    """
    The Grad-CAM test script @ Models
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='The Grad-CAM test script @ Models')

    # The arguments for model
    parser.add_argument('--pre_model', '-p', type=str,
                        default='./model_inference/cochldb.2.15.230316_v8_22050_Ensemble_Distill_cochldb.2.15.230316_v8_Ensemble_Distill_22050_model_pre.h5', 
                        help='model_pre path to load')
    parser.add_argument('--main_model', '-m', type=str, 
                        default='./model_inferences/cochldb.2.15.230316_v8_22050_Ensemble_Distill_cochldb.2.15.230316_v8_Ensemble_Distill_22050_result_230615-test_model_main.h5',
                        help='model_main path to load')
    
    # The arguments for audio source
    parser.add_argument('--audio_source', '-a', type=str, help='audio source path')
    parser.add_argument('--delay_time', '-dt', type=float, default=0.7, help='delay time')
    parser.add_argument('--front_time', '-ft', type=float, default=0.3, help='front time')
    parser.add_argument('--middle_time', '-mt', type=float, default=0.2, help='middle time')
    parser.add_argument('--class_name', '-c', type=str, default='Gunshot', help='original class name by sources')
    parser.add_argument('--expected_prob', '-ep', type=int, default=90, 
                        help='expected probability (%)')
    
    # The arguments for pass filtering
    parser.add_argument('--pf_type', '-pt', type=str, default='lpf', 
                        help='pass filter type')
    parser.add_argument('--pf_freq', '-pf', type=int, default=1000, 
                        help='pass filter frequency')
    parser.add_argument('--pf_rolloff', '-pr', type=float, default=0.5, 
                        help='pass filter rolloff')
    parser.add_argument('--output', '-o', type=str, default='gradcam.png', 
                        help='output path')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = cli()
    # 여기에서 args를 사용하여 다른 작업을 수행할 수 있습니다.
    print(args)