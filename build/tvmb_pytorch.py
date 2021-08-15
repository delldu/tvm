"""Onnx Model Tools."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 03月 23日 星期二 12:42:57 CST
# ***
# ************************************************************************************/
#
"""tvmb -- TVM Building From PyTorch ..."""

import numpy as np

import argparse
import pdb  # For debug
import os

import torch
import torch.nn as nn

import tvm
from tvm import relay, runtime, contrib


def save_tvm_model(relay_mod, relay_params, target, filename):
    # Create ouput file names
    file_name, _ = os.path.splitext(filename)
    so_filename = file_name + ".so"
    json_filename = file_name + ".json"
    params_filename = file_name + ".params"

    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(relay_mod, target=target, params=relay_params)

    lib.export_library(so_filename)

    with open(json_filename, "w") as f:
        f.write(graph)

    with open(params_filename, "wb") as f:
        f.write(runtime.save_param_dict(params))

    print("Building {} OK".format(file_name))


def load_tvm_model(filename, device):
    # Create input file names
    file_name, _ = os.path.splitext(filename)
    so_filename = file_name + ".so"
    json_filename = file_name + ".json"
    params_filename = file_name + ".params"

    graph = open(json_filename).read()
    loaded_solib = runtime.load_module(so_filename)
    loaded_params = bytearray(open(params_filename, "rb").read())

    mod = contrib.graph_executor.create(graph, loaded_solib, device)
    mod.load_params(loaded_params)

    return mod


def load_tvm_data(data_shape):
    np_data = np.random.uniform(size=data_shape).astype("float32")
    nd_data = tvm.nd.array(np_data, device)
    return nd_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="input traced model file (eg: mini.pt)")
    parser.add_argument("-b", "--build", help="build model", action="store_true")
    parser.add_argument("-v", "--verify", help="verify model", action="store_true")
    parser.add_argument("-g", "--gpu", help="use gpu", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    devname = "cuda" if args.gpu else "llvm"
    target = tvm.target.Target(devname, host="llvm --link-params=0")
    device = tvm.device(devname, 0)

    output_so_filename = "{}/{}_{}.so".format(args.output, "cuda" if args.gpu else "cpu", "mini")
    mini_model_filename = "{}/mini.pt".format(args.output)

    input_shape = (1, 3, 256, 256)

    # /************************************************************************************
    # ***
    # ***    Create Model
    # ***
    # ************************************************************************************/
    def create_model():
        class MiniModel(nn.Module):
            def __init__(self):
                """Init model."""
                super(MiniModel, self).__init__()
                self.conv3x3 = nn.Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.conv1x1 = nn.Conv2d(5, 5, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.relu = nn.ReLU()
                nn.init.kaiming_normal_(self.conv3x3.weight, mode="fan_out", nonlinearity="relu")
                nn.init.kaiming_normal_(self.conv1x1.weight, mode="fan_out", nonlinearity="relu")

            def forward(self, x):
                x = self.conv3x3(x)
                x = self.relu(x)
                x = self.conv1x1(x)
                x = self.relu(x)
                return x

        model = MiniModel()
        model = model.eval()
        traced_model = torch.jit.trace(model, torch.randn(input_shape))
        traced_model.save(mini_model_filename)

    # /************************************************************************************
    # ***
    # ***    Build Mdel
    # ***
    # ************************************************************************************/
    def build_model():
        print("Building model on {} ...".format(target))

        traced_model = torch.jit.load(args.input)
        print(traced_model.graph)
        mod, params = relay.frontend.from_pytorch(traced_model, [("input", input_shape)])

        save_tvm_model(mod, params, target, output_so_filename)

    # /************************************************************************************
    # ***
    # ***    Verify Mdel
    # ***
    # ************************************************************************************/
    def verify_model():
        print("Running model on {} ...".format(device))

        input_data = torch.randn(input_shape)
        nd_data = tvm.nd.array(input_data.numpy(), device)

        # Load module
        mod = load_tvm_model(output_so_filename, device)

        # TVM Run
        mod.set_input("input", nd_data)
        mod.run()
        output_data = mod.get_output(0)
        print(output_data)

        print("Evaluating ...")
        ftimer = mod.module.time_evaluator("run", device, number=2, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for ms
        print("Mean running time: %.2f ms (stdv: %.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

        traced_model = torch.jit.load(args.input)
        traced_model = traced_model.eval()
        with torch.no_grad():
            traced_output = traced_model(input_data)

        np.testing.assert_allclose(
            output_data.numpy(), traced_output.numpy(), rtol=1e-03, atol=1e-03
        )
        print("Running model OK.")

    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    if not os.path.exists(mini_model_filename):
        create_model()

    if args.build:
        build_model()

    if args.verify:
        verify_model()
