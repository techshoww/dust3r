import onnx
from onnx.shape_inference import infer_shapes
import onnxsim
import torch
from dust3r.model import AsymmetricCroCo3DStereo

def export_onnx(onnx_path):
    device=torch.device("cuda:0")
    weights_path = "DUSt3R_ViTLarge_BaseDecoder_512_linear.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
    model.forward = model.forward_onnx
    model.eval()
   
    img1 = torch.rand([1,3,384,512]).to(device)
    img2 = torch.rand([1,3,384,512]).to(device)
    inputs = (img1, img2)
    torch.onnx.export(model, inputs, onnx_path, input_names=['img1', 'img2',  ], output_names=["res1_pts3d", "res1_conf", "res2_pts3d_in_other_view", "res2_conf"], opset_version=12)

    onnx_model = onnx.load(onnx_path)
    onnx_model = infer_shapes(onnx_model)
    # convert model
    model_simp, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)
    print("onnx simpilfy successed, and model saved in {}".format(onnx_path))


if __name__=="__main__":
    export_onnx("dust3r.onnx")