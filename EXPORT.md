本文档描述dust3r模型的导出过程。

## 1. 从github clone源码  
```
git clone --recursive https://github.com/naver/dust3r
```

## 2. 新建export分支   
```
git checkout -b export  
```

## 3. 安装python环境  

```
conda create -n croco  python=3.10.15 -y
conda install pytorch torchvision -c pytorch
pip install onnx==1.17.0 onnxruntime==1.20.1 onnx-simplifier==0.4.36
pip install roma tqdm scipy huggingface_hub einops opencv-python  matplotlib
```

## 4. 下载pth模型  

```
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth
```

## 5. 修改torch模型代码以便导出和编译onnx  
完整修改可以通过git commit 记录查看  
(1). torch.expm1 算子替换  
```
def signed_expm1(x):
    sign = torch.sign(x)
    # return sign * torch.expm1(torch.abs(x))
    return sign * (torch.exp(torch.abs(x))-1)
```

(2). 固定算子输出固化到onnx模型  
同时可以避免导出 torch.cartesian_prod 算子  
```
self.pos = PositionGetter()(2,24,32, "cpu")
```

(3). 修改模型推理代码  


## 6. 导出 onnx 

(1). 导出onnx 
```
python export_onnx.py 
```

(2). 测试onnx
```
python infer_onnx.py
```

(3). 对比pth结果  
```
python infer.py
```

## 7. 编译onnx  
进入工具链环境，执行  
```
bash build.sh
```

## 8. 开发板运行axmodel  
```
ax_run_model -m dust3r.axmodel -w 10 -r 100
```
```
   Run AxModel:
         model: dust3r.axmodel
          type: 3 Core
          vnpu: Disable
      affinity: 0b001
        warmup: 10
        repeat: 100
         batch: { auto: 0 }
      parallel: false
   pulsar2 ver: 4.0 ce2fb6a4
    engine ver: 2.12.0s
      tool ver: 2.5.1a
      cmm size: 601133021 Bytes
  ---------------------------------------------------------------------------
  min = 319.663 ms   max = 320.067 ms   avg = 319.857 ms  median = 319.856 ms
   5% = 319.711 ms   90% = 319.953 ms   95% = 320.022 ms     99% = 320.067 ms
  ---------------------------------------------------------------------------
```