from init import *

def load_model():
    args = cli.cli()

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

    return model_pre, model_main, inner_model_2, inner_reduce_model_0,inner_reduce_model_1, inner_model_0, inner_model_1



def mel_spec_with_librosa(audio_path):
    y, sr = librosa.load(audio_path, sr=22050) # 로드 시 샘플링 레이트를 22050으로 설정
    frames = librosa.util.frame(y, frame_length=22050, hop_length=22050).T # 1초 단위로 나눔
    frames_reshaped = frames.reshape(frames.shape[0], 1, 22050) # 모델 입력을 위해 적절한 형태로 변환
    return y, sr, frames_reshaped

def load_class_list(model):
    class_cnt = model.get_layer('activation').get_weights()[0]
    
    if class_cnt == 752:
        tags = pd.read_csv('./model_inferences/cochldb.tags.csv')

    elif class_cnt == 772:
        tags = pd.read_csv('./model_inferences/cochldb.tags.csv')

    elif class_cnt == 104:
        tags = pd.read_csv('./model_inferences/esc50.tags.csv')

    else: 
        print("class amount vacancy.")

def find_class_num(tag_name, tags_df):
    class_num = tags_df[tags_df['tags'] == tag_name].index[0]
    return class_num


def predict(model_pre, model_main, tags, frames_reshaped):

    args = cli.cli()

    # 모델에 입력
    most_predicted = []
    target_predicted = []
    model_outputs = []
    class_num = find_class_num(args.class_name, tags) 


    # 0th : model_pre
    model_pre_output = model_pre.predict(frames_reshaped)

    # 1st : model_main
    # target tag probability (want to be predicted)
    for i in range (len(model_pre_output)):
        predicted_labels = model_main.predict(model_pre_output[i])
        target_tag_prob = predicted_labels[i][class_num]

    # highest probability tag (result)
        highest_prob = predicted_labels.max()
        pred_class_num = predicted_labels.argmax()
        pred_class_name = tags['tags'][pred_class_num] 

    # ------------------ TARGET RESULT ------------------
    print("="*20 + "RESULT" + "="*20)
    print(f"probability         | {target_tag_prob}")
    print(f"target_class_number | {class_num}")
    print(f"target_class        | {args.class_name}")
    print("="* 46)

    # ------------------ PREDICTED RESULT ------------------
    print("="*20 + "RESULT" + "="*20)
    print(f"probability            | {highest_prob}")
    print(f"predicted_class_number | {pred_class_num}")
    print(f"predicted_class        | {tags['tags'][pred_class_num]}")
    print("="* 46)

    return predicted_labels, target_tag_prob, highest_prob, pred_class_num, pred_class_name


def gradcam_result(model_pre_output, model, class_num = 53):
    grids = []
    grids1 = []

    # GradCAM을 적용합니다.
    for i in range(model_pre_output.shape[0]):
        
        model_data = np.expand_dims(model_pre_output[i], axis=0)

        # Create GradCAM
        if model.shape == 1:
            gradcamplpl = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
            gradcamplpl1 = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
        
        elif model.shape > 1:
            gradcamplpl = Gradcam(model[0], model_modifier=ReplaceToLinear(), clone=True)
            gradcamplpl1 = Gradcam(model[1], model_modifier=ReplaceToLinear(), clone=True)

        # Generate heatmap with GradCAM++
        data = (model_data, None)
        cam = gradcamplpl(CategoricalScore(class_num), model_data)
        cam1 = gradcamplpl1(CategoricalScore(class_num), model_data)

        # Render
        grids.append(cam)
        grids1.append(cam1)
    
    return grids, grids1

def tensorboard_vis():
    os.system('tensorboard --logdir ./logs/fit --host localhost --port 8088')
    


def plot_grad_cam(model_pre_output, grids, grids1):

    fig, axs = plt.subplots(2,5, figsize=(50, 35), facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)


    for i in range(model_pre_output.shape[0]):

        #grids[i] = grids[i].reshape((128, 96))
        #grids1[i] = grids1[i].reshape((128, 96))
        if i < 5:  
            axs[0][i].imshow(model_pre_output[i], aspect='auto', origin='lower')
            #axs[0][i].imshow(grids[i], alpha=0.5, aspect='auto', origin='lower')
            #axs[0][i].imshow(grids1[i], alpha=0.5, aspect='auto', origin='lower')
            axs[0][i].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)
            axs[0][i].set_ylabel('Mel Frequency', fontsize=35)
            axs[0][i].set_xlabel(f'Time Frame : {i+1}sec', fontsize=35)


        elif i >= 5:
            axs[1][i-5].imshow(model_pre_output[i], aspect='auto', origin='lower')
            #axs[1][i-5].imshow(grids[i], alpha=0.5, aspect='auto', origin='lower')
            #axs[1][i-5].imshow(grids1[i], alpha=0.5, aspect='auto', origin='lower')
            axs[1][i-5].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)
            axs[1][i-5].set_ylabel('Mel Frequency', fontsize=35)
            axs[1][i-5].set_xlabel(f'Time Frame: {i+1}sec', fontsize=35)

    plt.show()
        


def plot_activation_histograms(grids, predicted_labels, mode=0):
    '''
    mode = 0 : Time Frame
    mode = 1 : Mel Frequency
    '''
    fig, axs = plt.subplots(2, 5, figsize=(50, 35), facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)

    for i in range(len(grids)):
        # Red 채널의 평균 activation 값을 계산합니다.
        mean_activations_red = grids[i][:, :, 0].mean(axis=mode)

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

        # 축, 제목을 설정합니다.
        axs[i // 5, i % 5].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)

    plt.show()


def plot_activation_histograms_grad(grids, predicted_labels, mode=0):
    '''
    mode = 0 : Time Frame
    mode = 1 : Mel Frequency
    '''
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

        # 축, 제목을 설정합니다.
        axs[i // 5, i % 5].set_title(f'Probability : {predicted_labels.max(axis=1)[i]:.4f}', fontsize=35)

    plt.show()

