import torch
import sys

def pt2onnx(pt_path, onnx_path, device, input_size=(3, 200, 200)):
    model = torch.load(pt_path, map_location=device)
    model.eval()
    tracer_input = torch.randn(1, *input_size).to(device)
    torch.onnx.export(model, tracer_input, onnx_path)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Please specify the input file (.pt file to be converted).')
        print('Please specify where to save the output file (.onnx file).')
    # converting
    print('Converting...')
    pt2onnx(sys.argv[1], sys.argv[2], torch.device('cpu'))
    print('Succeeded!')
