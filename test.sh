#!/usr/bin/env bash

# # FL (512 256 128)
# python test.py --weights ./weights/20260127_121759/best_resnet34_global.pt --device-id 0 --resolution 576
# python test.py --weights ./weights/20260127_121759/best_resnet34_global.pt --device-id 0 --resolution 512
# python test.py --weights ./weights/20260127_121759/best_resnet34_global.pt --device-id 0 --resolution 384
# python test.py --weights ./weights/20260127_121759/best_resnet34_global.pt --device-id 0 --resolution 256
# python test.py --weights ./weights/20260127_121759/best_resnet34_global.pt --device-id 0 --resolution 192
# python test.py --weights ./weights/20260127_121759/best_resnet34_global.pt --device-id 0 --resolution 128

# # FL+MRKD (512 256 128)
# python test.py --weights ./weights/20260127_124135/best_resnet34_global.pt --device-id 0 --resolution 576
# python test.py --weights ./weights/20260127_124135/best_resnet34_global.pt --device-id 0 --resolution 512
# python test.py --weights ./weights/20260127_124135/best_resnet34_global.pt --device-id 0 --resolution 384
# python test.py --weights ./weights/20260127_124135/best_resnet34_global.pt --device-id 0 --resolution 256
# python test.py --weights ./weights/20260127_124135/best_resnet34_global.pt --device-id 0 --resolution 192
# python test.py --weights ./weights/20260127_124135/best_resnet34_global.pt --device-id 0 --resolution 128

# # FL (512 384 256)
# python test.py --weights ./weights/20260128_051904/best_resnet34_global.pt --device-id 0 --resolution 576
# python test.py --weights ./weights/20260128_051904/best_resnet34_global.pt --device-id 0 --resolution 512
# python test.py --weights ./weights/20260128_051904/best_resnet34_global.pt --device-id 0 --resolution 384
# python test.py --weights ./weights/20260128_051904/best_resnet34_global.pt --device-id 0 --resolution 256
# python test.py --weights ./weights/20260128_051904/best_resnet34_global.pt --device-id 0 --resolution 192
# python test.py --weights ./weights/20260128_051904/best_resnet34_global.pt --device-id 0 --resolution 128

# # FL+MRKD (512 384 256)
# python test.py --weights ./weights/20260128_052100/best_resnet34_global.pt --device-id 0 --resolution 576
# python test.py --weights ./weights/20260128_052100/best_resnet34_global.pt --device-id 0 --resolution 512
# python test.py --weights ./weights/20260128_052100/best_resnet34_global.pt --device-id 0 --resolution 384
# python test.py --weights ./weights/20260128_052100/best_resnet34_global.pt --device-id 0 --resolution 256
# python test.py --weights ./weights/20260128_052100/best_resnet34_global.pt --device-id 0 --resolution 192
# python test.py --weights ./weights/20260128_052100/best_resnet34_global.pt --device-id 0 --resolution 128

# CL (512 256 128)
python test.py --weights ./weights/20260128_114239/best_resnet34.pt --device-id 0 --resolution 576
python test.py --weights ./weights/20260128_114239/best_resnet34.pt --device-id 0 --resolution 512
python test.py --weights ./weights/20260128_114239/best_resnet34.pt --device-id 0 --resolution 384
python test.py --weights ./weights/20260128_114239/best_resnet34.pt --device-id 0 --resolution 256
python test.py --weights ./weights/20260128_114239/best_resnet34.pt --device-id 0 --resolution 192
python test.py --weights ./weights/20260128_114239/best_resnet34.pt --device-id 0 --resolution 128

# CL (512 384 256)
python test.py --weights ./weights/20260128_114422/best_resnet34.pt --device-id 0 --resolution 576
python test.py --weights ./weights/20260128_114422/best_resnet34.pt --device-id 0 --resolution 512
python test.py --weights ./weights/20260128_114422/best_resnet34.pt --device-id 0 --resolution 384
python test.py --weights ./weights/20260128_114422/best_resnet34.pt --device-id 0 --resolution 256
python test.py --weights ./weights/20260128_114422/best_resnet34.pt --device-id 0 --resolution 192
python test.py --weights ./weights/20260128_114422/best_resnet34.pt --device-id 0 --resolution 128
