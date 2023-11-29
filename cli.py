import argparse

def cli():
    parser = argparse.ArgumentParser(description='The Grad-CAM test script @ Models')

    # The arguments for model
    parser.add_argument('--pre_model', '-p', type=str,
                        default= './model_inference/cochldb.2.15.230316_v8_22050_Ensemble_Distill_cochldb.2.15.230316_v8_Ensemble_Distill_22050_model_pre.h5', 
                        help='model_pre path to load')
    parser.add_argument('--main_model', '-m', type=str, 
                        default='./model_inferences/cochldb.2.15.230316_v8_22050_Ensemble_Distill_cochldb.2.15.230316_v8_Ensemble_Distill_22050_result_230615-test_model_main.h5',
                        help='model_main path to load')
    parser.add_argument('--ensemble', '-e', type=str, default=True, 
                        help='true or false which is ensemble model')
    
    # The arguments for audio source
    parser.add_argument('--audio_source', '-a', type=str, help='audio source path')
    parser.add_argument('--delay_time', '-d', type=float, default=0.7, help='delay time')
    parser.add_argument('--front_time', '-f', type=float, default=0.3, help='front time')
    parser.add_argument('--middle_time', '-i', type=float, default=0.2, help='middle time')
    parser.add_argument('--class_name', '-c', type=str, default='gunshot', help='original class name by sources')
    parser.add_argument('--expected_prob', '-e', type=int, default=90, 
                        help='expected probability (%)')
    
    # The arguments for pass filtering
    parser.add_argument('--pf_type', '-t', type=str, default='lpf', 
                        help='pass filter type')
    parser.add_argument('--pf_freq', '-q', type=int, default=1000, 
                        help='pass filter frequency')
    parser.add_argument('--pf_rolloff', '-r', type=float, default=0.5, 
                        help='pass filter rolloff')
    parser.add_argument('--output', '-o', type=str, default='gradcam.png', 
                        help='output path')
    
    return parser.parse_args()

if __name__ == '__main__':
    cli()
    
