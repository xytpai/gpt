import os
import torch


WEIGHT_DIR_NAME = './checkpoints'


def load_model(args, model):
    def get_milestone(fname):
        return int(fname.split('_')[1].replace('.ckpt', ''))
    files = []
    load_dir = args.load if len(args.load) > 0 else WEIGHT_DIR_NAME
    for path, dir_list, file_list in os.walk(load_dir):
        for file in file_list:
            if file.endswith('.ckpt') and file.startswith(args.model):
                files.append(file)
    files = sorted(files, key=lambda x : get_milestone(x), reverse=True)
    if len(files) > 0:
        args.begin = get_milestone(files[0])
        ckpt = torch.load(os.path.join(path, files[0]), map_location='cpu')
        ckpt_model = ckpt.get('model', None)
        if ckpt_model is None:
            missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(ckpt_model, strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print('load model: ' + str({'missing_keys':missing_keys, 'unexpected_keys':unexpected_keys}))
        args.ckpt = ckpt
    return
