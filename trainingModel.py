import os
import argparse
import torch
from learnedMethodForHologram.watermelon_hologram.watermelon import watermelon_without_GAN as watermelon
from learnedMethodForHologram.watermelon_hologram.data_loader import dataloaderImgDepthAmpPhs as data_loader
from learnedMethodForHologram import utilities

def check_and_create_folder(path):
    if not os.path.exists(path):
        print(f"Folder {path} does not exist, creating it...")
        os.makedirs(path)

def train_gan(train_img_path, train_depth_path, train_amp_path, train_phs_path,
              validate_img_path, validate_depth_path, validate_amp_path, validate_phs_path,
              samplesNum, channlesNum, height, width, batch_size, lr_G, lr_D, epoch_num,
              save_path_G, save_path_D, loss_metrics_file, save_path_img):
    
    utilities.set_seed(122731)
    
    dataset_train = data_loader(
        img_path=train_img_path,
        depth_path=train_depth_path,
        amp_path=train_amp_path,
        phs_path=train_phs_path,
        samplesNum=samplesNum,
        channlesNum=channlesNum,
        height=height,
        width=width,
        cuda=True,
    )
    
    dataLoader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    
    dataset_validate = data_loader(
        img_path=validate_img_path,
        depth_path=validate_depth_path,
        amp_path=validate_amp_path,
        phs_path=validate_phs_path,
        samplesNum=100,
        channlesNum=channlesNum,
        height=height,
        width=width,
        cuda=True,
    )
    
    dataLoader_validate = torch.utils.data.DataLoader(
        dataset_validate,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=0,
    )
    
    GAN = watermelon(
        filter_radius_coefficient=0.45,
        pad_size=320,
        distance_stack=torch.linspace(-4e-4, 0.0, 21)[:-1],
        pretrained_model_path_G=None,
        pretrained_model_path_D=None,
        input_shape=(1, 4, height, width),
        cuda=True,
    )
    
    check_and_create_folder(os.path.dirname(save_path_G))
    check_and_create_folder(os.path.dirname(save_path_D))
    check_and_create_folder(os.path.dirname(loss_metrics_file))
    check_and_create_folder(save_path_img)
    
    GAN.train(
        data_loader_train=dataLoader_train,
        data_loader_val=dataLoader_validate,
        phs_gradient_loss_weight=1,
        perceptual_loss_weight=1e-1,
        pixel_loss_weight=1,
        TV_loss_weight=1e-3,
        discriminator_loss_weight=1e-1,
        epoch_num=epoch_num,
        lr_G=lr_G,
        lr_D=lr_D,
        save_path_G=save_path_G,
        save_path_D=save_path_D,
        info_print_interval=50,
        info_plot_interval=50,
        loss_metrics_file=loss_metrics_file,
        save_path_img=save_path_img,
        checkpoint_iterval=1,
        discriminator_train_ratio=5,
        discriminator_lambda=10,
        step_scheduler_G_gamma=0.9999,
        step_scheduler_D_gamma=0.9999,
        visualization_RGBD_AP=dataset_validate[0],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN model for hologram generation.")
    
    # required arguments
    parser.add_argument('--train_img_path', type=str, required=True, help="Path to training image binary file.")
    parser.add_argument('--train_depth_path', type=str, required=True, help="Path to training depth binary file.")
    parser.add_argument('--train_amp_path', type=str, required=True, help="Path to training amplitude binary file.")
    parser.add_argument('--train_phs_path', type=str, required=True, help="Path to training phase binary file.")
    
    parser.add_argument('--validate_img_path', type=str, required=True, help="Path to validation image binary file.")
    parser.add_argument('--validate_depth_path', type=str, required=True, help="Path to validation depth binary file.")
    parser.add_argument('--validate_amp_path', type=str, required=True, help="Path to validation amplitude binary file.")
    parser.add_argument('--validate_phs_path', type=str, required=True, help="Path to validation phase binary file.")
    
    parser.add_argument('--samplesNum', type=int, required=True, help="Number of samples in the dataset.")
    parser.add_argument('--channlesNum', type=int, required=True, help="Number of channels.")
    parser.add_argument('--height', type=int, required=True, help="Image height.")
    parser.add_argument('--width', type=int, required=True, help="Image width.")
    
    # optional arguments
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training. Default is 4.")
    parser.add_argument('--lr_G', type=float, default=1e-3, help="Learning rate for generator. Default is 1e-3.")
    parser.add_argument('--lr_D', type=float, default=1e-3, help="Learning rate for discriminator. Default is 1e-3.")
    parser.add_argument('--epoch_num', type=int, default=50, help="Number of training epochs. Default is 50.")
    
    # output paths
    parser.add_argument('--save_path_G', type=str, required=True, help="Path to save the generator model.")
    parser.add_argument('--save_path_D', type=str, required=True, help="Path to save the discriminator model.")
    parser.add_argument('--loss_metrics_file', type=str, required=True, help="Path to save the loss metrics file.")
    parser.add_argument('--save_path_img', type=str, required=True, help="Path to save generated images.")
    
    args = parser.parse_args()
    
    train_gan(
        args.train_img_path, args.train_depth_path, args.train_amp_path, args.train_phs_path,
        args.validate_img_path, args.validate_depth_path, args.validate_amp_path, args.validate_phs_path,
        args.samplesNum, args.channlesNum, args.height, args.width, args.batch_size, args.lr_G, args.lr_D,
        args.epoch_num, args.save_path_G, args.save_path_D, args.loss_metrics_file, args.save_path_img
    )
