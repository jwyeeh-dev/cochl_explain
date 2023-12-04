# Cochl_CAM
This repository is the tool which can test gradient concentration by CAM on your models.


## Pre-requisites (Grad-CAM) 
```
! pip install -r requirements.txt
```

## Grad-CAM Testing Script (.ipynb)
This file have scripts about Grad-CAM testing, visualizing.
- **Model loading**
- **Grad-CAM testing**
- **Grad-CAM visualization functions**
 

## Grad-CAM Testing Tool (Python module)
This file is the tool which is working on 
- any model 
  - they have Activation function and Convolution layer
- any input length
  - they can split about 1sec
  - mp3, wav, etc.


### Working Command

```
$ python test.py -p [model_pre] -m [model_main] 
```

### CLI structures

```
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
    parser.add_argument('--class_list', '-cl', type=str,
                        help='class_list path to load')
    
    # The arguments for audio source
    parser.add_argument('--audio_source', '-a', type=str, help='audio source path')
    parser.add_argument('--class_name', '-c', type=str, default='Gunshot', help='original class name by sources')

    """
    The arguments for Modifying the audio source
    """
    # The arguments for audio preprocessing
    parser.add_argument('--delay_time', '-dt', type=float, default=0.7, help='delay time')
    parser.add_argument('--front_time', '-ft', type=float, default=0.3, help='front time')
    parser.add_argument('--middle_time', '-mt', type=float, default=0.2, help='middle time')
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
```


### Supported visualization methods 

- GradCAM (tf-keras-vis, tf-explain)
- Single Plot
- Subplot according to time range
- Distribution of concentrated gradient
- (to be contd.)

