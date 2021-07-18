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

    input_shape = (1, 3, 256, 512)

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

        # def @main(%input: Tensor[(1, 3, ?, ?), float32]) {
        #   %0 = nn.conv2d(%input, meta[relay.Constant][0], padding=[1, 1, 1, 1], kernel_size=[3, 3]);
        #   %1 = nn.bias_add(%0, meta[relay.Constant][1]);
        #   nn.relu(%1)
        # }

        with tvm.transform.PassContext(opt_level=3):
            vm_exec = relay.vm.compile(mod, target=target, params=params)

        code, lib = vm_exec.save()
        lib.export_library(so_path)
        with open(ro_path, "wb") as fo:
            fo.write(code)

        # (Pdb) print(vm_exec.bytecode)
        # VM Function[0]: main(input)
        # # reg file size = 15
        # # instruction count = 18
        # opcode, fields # inst(text):
        #  0: 11 0 1   # load_const $1 Const[0]
        #  1: 16 1 64 2 32 1 1 2   # alloc_storage $2 $1 64 float32 1
        # ==> alloc_storage(size, alignment, device, dtype_hint="float32"):

        #  2: 11 1 3   # load_const $3 Const[1]
        #  3: 5 2 3 2 32 1 5 4 1 1 512 512 3   # alloc_tensor $4 $2 $3 [1, 1, 512, 512, 3] float32
        # ==> alloc_tensor(storage, offset, shape=[1, 1, 512, 512, 3], dtype="float32", assert_shape=None)

        #  4: 4 0 2 1 0 4   # invoke_packed PackedFunc[0] (in: $0, out: $4)
        # ==> fused_layout_transform_2

        #  5: 11 2 5   # load_const $5 Const[2]
        #  6: 16 5 64 2 32 1 1 6   # alloc_storage $6 $5 64 float32 1
        # ==> alloc_storage(size, alignment, device, dtype_hint="float32"):

        #  7: 11 3 7   # load_const $7 Const[3]
        #  8: 5 6 7 2 32 1 5 8 1 1 512 512 8   # alloc_tensor $8 $6 $7 [1, 1, 512, 512, 8] float32
        # ==> alloc_tensor(storage, offset, shape=[1, 1, 512, 512, 8], dtype="float32", assert_shape=None)

        #  9: 11 4 9   # load_const $9 Const[4]
        # 10: 11 5 10   # load_const $10 Const[5]
        # 11: 4 1 4 1 4 9 10 8   # invoke_packed PackedFunc[1] (in: $4, $9, $10, out: $8)
        # 12: 11 6 11   # load_const $11 Const[6]
        # 13: 16 11 64 2 32 1 1 12   # alloc_storage $12 $11 64 float32 1
        # 14: 11 7 13   # load_const $13 Const[7]

        # 15: 5 12 13 2 32 1 4 14 1 8 512 512   # alloc_tensor $14 $12 $13 [1, 8, 512, 512] float32
        # ==> alloc_tensor(storage, offset, shape=[1, 8, 512, 512], dtype="float32", assert_shape=None)

        # 16: 4 2 2 1 8 14   # invoke_packed PackedFunc[2] (in: $8, out: $14)
        # ==> fused_layout_transform_3

        # 17: 1 14   # ret $14

        # (Pdb) print(vm_exec.stats)
        # Relay VM executable statistics:
        #   Constant shapes (# 8): [scalar, scalar, scalar, scalar, [1, 1, 3, 3, 3, 8], 
        # [1, 1, 1, 1, 8], 
        # scalar, scalar]
        #   Globals (#1): [("main", 0)]
        #   Primitive ops (#3): [fused_layout_transform_2, fused_nn_contrib_conv2d_NCHWc_add_nn_relu, 
        #  fused_layout_transform_3]


        print("Building model OK")

    def verify():
        """Verify model."""

        print("Running model on {} ...".format(device))

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
        create_micro()

    if not os.path.exists(args.input):
        print("ONNX model file {} not exist.".format(args.input))
        os._exit(0)

    if args.build:
        build()

    if args.verify:
        verify()
