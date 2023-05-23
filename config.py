params = dict()

params['num_classes'] = 101

params['dataset'] = 'D:/Projects/ai-research-school/datasets/UCF101'
#params['dataset'] = '/home/Dataset/hmdb'

params['epoch_num'] = 300  # 600
params['batch_size'] = 8
params['epoch_checkpoints'] = 20
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 0.01  # 0.001
params['momentum'] = 0.9
params['weight_decay'] = 0.0005
params['display'] = 100  # 1000
params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'log'
#params['save_path'] = 'UCF101'
params['save_path_base']='D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/'
params['data']='UCF-101'
