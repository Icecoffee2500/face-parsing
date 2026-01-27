# no transform + don't ignore background (SegFaceCeleb)
python train.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name no-transform-background_on-resnet34_warmup_1_bs32_gpu3 --lr-warmup-epochs 1 --device-id 0

# no transform (SegFaceCeleb)
python train.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name no-transform-SegFaceCeleb-resnet34_warmup_1_bs32 --lr-warmup-epochs 1 --ignore-background --device-id 0

# all transform (SegFaceCeleb)
python train.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name all-transform-SegFaceCeleb-resnet34_warmup_1_bs32 --lr-warmup-epochs 1 --ignore-background --device-id 1

# RandomScale, RandomCrop 제외 (SegFaceCeleb)
python train.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name no-crop_scale-SegFaceCeleb-resnet34_warmup_1_bs32 --lr-warmup-epochs 1 --ignore-background --device-id 3

# no transform (BiSeNetCeleb)
python train.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name no-transform-BiSeNetCeleb-resnet34_warmup_1_bs32 --lr-warmup-epochs 1 --ignore-background --device-id 0

# [FL] no transform + don't ignore background (SegFaceCeleb)
python train_fl.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name fl-no-transform-background_on-resnet34_warmup_1_bs32_gpu3 --lr-warmup-epochs 1 --device-id 2

# [FL] no transform + don't ignore background (SegFaceCeleb) with Resolutions (512, 256, 128)
python train_fl.py --batch-size 32 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name fl-no-transform-background_on-resnet34_warmup_1_bs32_gpu3_resolutions_512_256_128 --lr-warmup-epochs 1 --client-resolutions 512 256 128 --device-id 0

# [FL] no transform + don't ignore background (SegFaceCeleb) with Resolutions (512, 256, 128) and MRKD
python train_fl.py --batch-size 8 --wandb --wandb-log-images --backbone resnet34 --wandb-run-name fl-no-transform-background_on-resnet34_warmup_1_bs32_gpu3_resolutions_512_256_128 --lr-warmup-epochs 1 --client-resolutions 512 256 128 --mrkd --device-id 0