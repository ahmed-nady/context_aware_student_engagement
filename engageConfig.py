import openai


openai.api_key = ""
#======action recognition config===#
pretrained_model_path  = './pretrained_action_recog_models/ckpt_epoch_30.pth'
config = './pretrained_action_recog_models/16_32_vifi_clip_16_shot_cvpr_n_actions.yaml'
labels_file = "action_labels.csv"
#======video Conf===#
v_path ="./videos/dataset_20230713-133150_0_S02_T76_E2.mp4"
clip_len = 15*3
#=====object detcetor conf===#
yolox_model = ""

ar_period= 15*3
engage_period = 15*120

