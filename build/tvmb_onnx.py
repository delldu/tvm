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
"""tvmb -- TVM Building Onnx ..."""

import numpy as np

import argparse
import pdb  # For debug
import os

import onnx
import onnxruntime

import tvm
from tvm import relay, runtime, contrib


def onnx_load(onnx_file):
    session_options = onnxruntime.SessionOptions()
    # session_options.log_severity_level = 0

    # Set graph optimization level
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    )

    onnx_model = onnxruntime.InferenceSession(onnx_file, session_options)
    # onnx_model.set_providers(['CUDAExecutionProvider'])
    print(
        "Onnx Model Engine: ",
        onnx_model.get_providers(),
        "Device: ",
        onnxruntime.get_device(),
    )

    return onnx_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="input onnx file (eg: mini.onnx)")
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

    onnx_so_path = "{}/{}_{}.so".format(args.output, devname, os.path.basename(args.input))
    onnx_json_path = "{}/{}_{}.json".format(args.output, devname, os.path.basename(args.input))
    onnx_params_path = "{}/{}_{}.bin".format(args.output, devname, os.path.basename(args.input))

    onnx_input_shape = (1, 3, 256, 256)
    onnx_shape_dict = {"input": onnx_input_shape}

    # /************************************************************************************
    # ***
    # ***    Create Model
    # ***
    # ************************************************************************************/
    def create_model():
        import torch
        import torch.nn as nn

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

        def onnx_export():
            model = MiniModel()
            model = model.eval()

            x = torch.randn(onnx_input_shape)
            # with torch.no_grad():
            #     y = model(x)

            torch.onnx.export(
                model,
                x,
                "mini.onnx",
                input_names=["input"],
                output_names=["output"],
                verbose=True,
                opset_version=11,
                keep_initializers_as_inputs=False,
                export_params=True,
            )

        onnx_export()

    # /************************************************************************************
    # ***
    # ***    Build Mdel
    # ***
    # ************************************************************************************/
    def build_model():
        print("Building model on {} ...".format(target))

        onnx_model = onnx.load(args.input)

        mod, params = relay.frontend.from_onnx(
            onnx_model, shape=onnx_shape_dict, freeze_params=True
        )
        print(mod)

        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)

        lib.export_library(onnx_so_path)

        with open(onnx_json_path, "w") as json_file:
            json_file.write(graph)

        with open(onnx_params_path, "wb") as params_file:
            params_file.write(runtime.save_param_dict(params))

        print("Building model OK")

    # /************************************************************************************
    # ***
    # ***    Verify Mdel
    # ***
    # ************************************************************************************/
    def verify_model():
        print("Running model on {} ...".format(device))

        np_data = np.random.uniform(size=onnx_input_shape).astype("float32")
        nd_data = tvm.nd.array(np_data, device)

        # Load module
        graph = open(onnx_json_path).read()
        loaded_solib = runtime.load_module(onnx_so_path)
        loaded_params = bytearray(open(onnx_params_path, "rb").read())

        mod = contrib.graph_executor.create(graph, loaded_solib, device)
        mod.load_params(loaded_params)

        # TVM Run
        mod.set_input("input", nd_data)
        mod.run()
        output_data = mod.get_output(0)
        print(output_data)

        print("Evaluating ...")
        ftimer = mod.module.time_evaluator("run", device, number=2, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for ms
        print("Mean running time: %.2f ms (stdv: %.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

        onnxruntime_engine = onnx_load(args.input)
        onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: np_data}
        onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)

        np.testing.assert_allclose(
            output_data.numpy(), onnxruntime_outputs[0], rtol=1e-03, atol=1e-03
        )
        print("Running model OK.")

    # /************************************************************************************
    # ***
    # ***    Flow Control
    # ***
    # ************************************************************************************/
    if not os.path.exists("mini.onnx"):
        create_model()

    if not os.path.exists(args.input):
        print("ONNX model file {} not exist.".format(args.input))
        os._exit(0)

    if args.build:
        build_model()

    if args.verify:
        verify_model()
