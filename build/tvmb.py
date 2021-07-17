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
from tvm import relay, runtime

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
        target = tvm.target.Target("cuda", host="llvm")
    else:
        target = tvm.target.Target("llvm", host="llvm")
    device = tvm.device(str(target), 0)

    if args.gpu:
        so_path = "{}/cuda_{}.so".format(args.output, os.path.basename(args.input))
        ro_path = "{}/cuda_{}.ro".format(args.output, os.path.basename(args.input))
    else:
        so_path = "{}/cpu_{}.so".format(args.output, os.path.basename(args.input))
        ro_path = "{}/cpu_{}.ro".format(args.output, os.path.basename(args.input))

    input_shape = (1, 3, 512, 512)
    # input_shape = (1, 3, relay.Any(), relay.Any())

    def micro_onnx():
        import torch
        import torch.nn as nn

        class MicroModel(nn.Module):
            def __init__(self):
                """Init model."""
                super(MicroModel, self).__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                return x

        def onnx_export():
            model = MicroModel()
            model = model.eval()

            x = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                y = model(x)

            dynamic_axes = {
                "input": {2: "height", 3: "width"},
            }

            torch.onnx.export(
                model,
                x,
                "micro.onnx",
                input_names=["input"],
                output_names=["output"],
                verbose=True,
                opset_version=11,
                keep_initializers_as_inputs=False,
                dynamic_axes=dynamic_axes,
                export_params=True,
            )

        onnx_export()

    def build():
        """Building model."""

        print("Building model on {} ...".format(target))

        onnx_model = onnx.load(args.input)

        # Parsing onnx model
        # mod, params = relay.frontend.from_onnx(
        #     onnx_model, {"input": input_shape}, freeze_params=True
        # )
        mod, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
        print(mod)

        with tvm.transform.PassContext(opt_level=3):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

        code, lib = vm_exec.save()
        lib.export_library(so_path)
        with open(ro_path, "wb") as fo:
            fo.write(code)

        print("Building model OK")

    def verify():
        """Verify model."""

        print("Running model on {} ...".format(device))

        input_shape = [1, 3, 512, 512]
        np_data = np.random.uniform(size=input_shape).astype("float32")
        nd_data = tvm.nd.array(np_data, device)

        loaded_lib = runtime.load_module(so_path)
        loaded_code = bytearray(open(ro_path, "rb").read())
        vm_exec = runtime.vm.Executable.load_exec(loaded_code, loaded_lib)
        vm = runtime.vm.VirtualMachine(vm_exec, device)

        output_data = vm.run(nd_data)

        print(output_data.numpy())
        print("Running model OK.")

        print("Evaluating ...")
        ftimer = vm.module.time_evaluator("invoke", device, number=2, repeat=10)
        prof_res = np.array(ftimer("main", nd_data).results) * 1000  # multiply 1000 for ms
        print(
            "Mean running time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
        )

    if not os.path.exists("micro.onnx"):
        micro_onnx()

    if not os.path.exists(args.input):
        print("ONNX model file {} not exist.".format(args.input))
        os._exit(0)

    if args.build:
        build()

    if args.verify:
        verify()
