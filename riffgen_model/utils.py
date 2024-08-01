import torch


class config:
    batch_size = 1
    lr = 1e-4
    grad_acc_step = 4
    max_duration = 30
    sr = 44100
    epochs = 50
    split = 100
    target_size = 1323000
    audio_length = 2000
    mini_dataset_id = "lewtun/music_genres"
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_outpath = "riffgen"
    model_filename = "riffgen.pth"
    hf_model_id = "tensorkelechi/riffgen"
