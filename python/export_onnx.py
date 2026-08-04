import os
import sys
import torch

sys.path.extend(['.', 'build', '../build', os.path.join(os.getcwd(), 'build')])
from network import ZeroCrossNet

def export_to_onnx():
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    pth_path = os.path.join(models_dir, 'best_model.pth')
    onnx_path = os.path.join(models_dir, 'best_model.onnx')

    if not os.path.exists(pth_path):
        print(f"Error: Could not find {pth_path}")
        return

    print("Loading PyTorch model...")
    net = ZeroCrossNet()
    
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    
    net.eval()

    dummy_input = torch.randn(1, 6, 9, 9)

    print(f"Exporting to {onnx_path}...")
    torch.onnx.export(
        net,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy_logits', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'policy_logits': {0: 'batch_size'}, 
            'value': {0: 'batch_size'}
        }
    )
    print("Export complete!")

if __name__ == "__main__":
    export_to_onnx()