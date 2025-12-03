""" Load a checkpoint path and convert to ONNX model """
import sys
import os
import torch
import torch.nn as nn
import numpy as np

def main():
    try:
        fpath = sys.argv[1]
    except IndexError:
        print("arg: path to checkpoint file")
        sys.exit(-1)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{fpath}")

    ckpt = torch.load(fpath, map_location="cpu", weights_only=True)

    input_dim, hidden_dim, output_dim, dropout_rate = [ckpt[k] for k in
         ["input_dim","hidden_dim","output_dim","dropout_rate"]]

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dynamic_shapes = ({0: "batch_size"},)

    # NOTE: Requires torch >= 2.6
    example_input = torch.rand(1,input_dim)
    onxp = torch.onnx.export(
        model, 
        example_input,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
        opset_version=21)

    root, ext = os.path.splitext(fpath)
    output = root + ".onnx"
    onxp.save(output)

if __name__ == "__main__":
    main()
