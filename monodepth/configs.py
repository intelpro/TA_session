import argparse

def get_config(params):
    params["data_dir"] = "/content/gdrive/My Drive/samsung_TAsession/2011_09_26_drive_0096_sync"
    params["model_path"] = "./saved_ckpt"
    params["output_directory"] = None
    params["model"] = 'resnet18_md'
    params["mode"] = 'train'
    params["pretrained"] = False
    params["batch_size"] = 20
    params["epochs"] = 50
    params["learning_rate"] = 1e-4
    params["adjust_lr"] = True
    params["device"] = 'cuda:0'
    params["do_augmentation"] = True
    params["augment_parameters"]  = [0.8, 1.2 , 0.5, 2.0, 0.8, 1.2] 
    params["print_image"] = False
    params["print_weights"] = False
    params["input_channels"] = 3
    params["num_workers"] = 4
    params["use_multiple_gpu"] = False
    params["input_width"] = 512
    params["input_height"] = 256
    params["n_scale"] = 4
    params["SSIM_w"] = 0.85
    params["disp_grad_w"] = 0.1
    params["lr_w"] = 1
    return params
