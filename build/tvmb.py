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
"""tvmb -- TVM Building System ..."""

import numpy as np

import argparse
import pdb  # For debug
import os
import onnx
import tvm
from tvm import relay, runtime, contrib

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", type=str, help="input onnx file (eg: micro.onnx)")
    parser.add_argument("-b", "--build", help="build model", action="store_true")
    parser.add_argument("-v", "--verify", help="verify model", action="store_true")
    parser.add_argument("-g", "--gpu", help="use gpu", action="store_true")
    parser.add_argument("-o", "--output", type=str, default="models", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # /************************************************************************************
    # ***
    # ***    MS: Define Global Names
    # ***
    # ************************************************************************************/
    if args.gpu:
        target = tvm.target.Target("cuda", host="llvm --runtime=c++ --system-lib")
    else:
        target = tvm.target.Target("llvm", host="llvm --runtime=c++ --system-lib")
    device = tvm.device(str(target), 0)

    if args.gpu:
        onnx_so_path = "{}/cuda_{}.so".format(args.output, os.path.basename(args.input))
        onnx_json_path = "{}/cuda_{}.json".format(args.output, os.path.basename(args.input))
        onnx_params_path = "{}/cuda_{}.bin".format(args.output, os.path.basename(args.input))
    else:
        onnx_so_path = "{}/cpu_{}.so".format(args.output, os.path.basename(args.input))
        onnx_json_path = "{}/cpu_{}.json".format(args.output, os.path.basename(args.input))
        onnx_params_path = "{}/cpu_{}.bin".format(args.output, os.path.basename(args.input))

    onnx_input_shape = (1, 3, 256, 256)

    def create_micro():
        import torch
        import torch.nn as nn

        class MicroModel(nn.Module):
            def __init__(self):
                """Init model."""
                super(MicroModel, self).__init__()
                self.conv3x3 = nn.Conv2d(3, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.conv1x1 = nn.Conv2d(5, 5, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv3x3(x)
                x = self.relu(x)
                x = self.conv1x1(x)
                x = self.relu(x)
                return x

        def onnx_export():
            model = MicroModel()
            model = model.eval()

            x = torch.randn(onnx_input_shape)
            # with torch.no_grad():
            #     y = model(x)

            torch.onnx.export(
                model,
                x,
                "micro.onnx",
                input_names=["input"],
                output_names=["output"],
                verbose=True,
                opset_version=11,
                keep_initializers_as_inputs=False,
                export_params=True,
            )

        onnx_export()

    def build():
        """Building model."""

        print("Building model on {} ...".format(target))

        onnx_model = onnx.load(args.input)

        mod, params = relay.frontend.from_onnx(
            onnx_model, shape={"input": onnx_input_shape}, freeze_params=True
        )
        print(mod)

        # def @main(%input: Tensor[(1, 3, ?, ?), float32]) {
        #   %0 = nn.conv2d(%input, meta[relay.Constant][0], padding=[1, 1, 1, 1], kernel_size=[3, 3]);
        #   %1 = nn.bias_add(%0, meta[relay.Constant][1]);
        #   nn.relu(%1)
        # }

        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            graph, lib, params = relay.build(mod, target=target, params=params)

        lib.export_library(onnx_so_path)

        with open(onnx_json_path, "w") as f_json:
            f_json.write(graph)

        with open(onnx_params_path, "wb") as f_bin:
            f_bin.write(runtime.save_param_dict(params))

        print("Building model OK")

    def verify():
        """Verify model."""

        print("Running model on {} ...".format(device))

        np_data = np.random.uniform(size=onnx_input_shape).astype("float32")
        nd_data = tvm.nd.array(np_data, device)

        # Load module
        loaded_json = open(onnx_json_path).read()
        loaded_solib = runtime.load_module(onnx_so_path)
        loaded_params = bytearray(open(onnx_params_path, "rb").read())

        mod = contrib.graph_executor.create(loaded_json, loaded_solib, device)
        mod.load_params(loaded_params)

        # Run
        mod.set_input("input", nd_data)
        mod.run()
        output_data = mod.get_output(0)

        print(output_data.numpy())
        print("Running model OK.")

        print("Evaluating ...")
        ftimer = mod.module.time_evaluator("run", device, number=2, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for ms
        print(
            "Mean running time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
        )

    if not os.path.exists("micro.onnx"):
        create_micro()

    if not os.path.exists(args.input):
        print("ONNX model file {} not exist.".format(args.input))
        os._exit(0)

    if args.build:
        build()

    if args.verify:
        verify()
