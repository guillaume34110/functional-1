"""Build browser ONNX assets from the frozen Functional 1 and 2 checkpoints."""
import argparse
from functools import reduce
import io
import json
from pathlib import Path
from shutil import copyfile

import onnx
from onnx import TensorProto, helper
import torch
from torch.onnx import symbolic_helper
from torch.onnx.symbolic_opset16 import scatter_reduce as default_scatter_reduce
from torch.onnx.symbolic_opset9 import flatten as default_flatten

from browser_assets import release_metadata
from functional_two import FunctionalTwoSession
from play import NeuralSession, RELEASE, version_page


OPSET = 18


def constant(graph, value):
    return graph.op("Constant", value_t=torch.tensor(value))


def membership(graph, values, choices, *_):
    count = symbolic_helper._get_tensor_sizes(choices)[0]
    equals = [graph.op("Equal", values, graph.op("Gather", choices, constant(graph, i), axis_i=0))
              for i in range(count)]
    return reduce(lambda a, b: graph.op("Or", a, b), equals)


def logaddexp(graph, a, b):
    maximum = graph.op("Max", a, b)
    left = graph.op("Exp", graph.op("Sub", a, maximum))
    right = graph.op("Exp", graph.op("Sub", b, maximum))
    combined = graph.op("Add", maximum, graph.op("Log", graph.op("Add", left, right)))
    return graph.op("Where", graph.op("IsInf", maximum), maximum, combined)


def expm1(graph, x):
    square = graph.op("Mul", x, x)
    coefficient = graph.op("Add", constant(graph, 0.5), graph.op("Mul", x, constant(graph, 1 / 6)))
    near_zero = graph.op("Add", x, graph.op("Mul", square, coefficient))
    other = graph.op("Sub", graph.op("Exp", x), constant(graph, 1.0))
    return graph.op("Where", graph.op("Less", graph.op("Abs", x), constant(graph, 0.001)), near_zero, other)


def flatten(graph, x, start, end):
    rank = symbolic_helper._get_tensor_rank(x)
    if not rank:
        return default_flatten(graph, x, start, end)
    first = symbolic_helper._parse_arg(start, "i") % rank
    last = symbolic_helper._parse_arg(end, "i") % rank + 1
    sizes = symbolic_helper._get_tensor_sizes(x)
    if first < 2 or (sizes and sizes[0] not in (0, None)):
        return default_flatten(graph, x, start, end)
    prefix = graph.op("Shape", x, end_i=first)
    middle = graph.op("Shape", x, start_i=first, end_i=last)
    width = graph.op("ReduceProd", middle, constant(graph, [0]), keepdims_i=1)
    dimensions = [prefix, width]
    if last < rank:
        dimensions.append(graph.op("Shape", x, start_i=last))
    return graph.op("Reshape", x, graph.op("Concat", *dimensions, axis_i=0), allowzero_i=1)


def scatter_reduce(graph, data, dim, indices, updates, reduction, include_self):
    if (symbolic_helper._parse_arg(dim, "i") == 1
            and symbolic_helper._parse_arg(reduction, "s") == "amax"
            and symbolic_helper._parse_arg(include_self, "b")):
        # Axis 1 cannot be scalar, so the export needs no scalar reshape/squeeze branches.
        return graph.op("ScatterElements", data, indices, updates, axis_i=1, reduction_s="max")
    return default_scatter_reduce(graph, data, dim, indices, updates, reduction, include_self)


def runtime_graph(model):
    model = onnx.shape_inference.infer_shapes(model)
    infos = (*model.graph.value_info, *model.graph.input, *model.graph.output)
    types = {value.name: value.type.tensor_type.elem_type for value in infos}
    types.update({value.name: value.data_type for value in model.graph.initializer})
    shapes = {value.name: tuple(dim.dim_value if dim.HasField("dim_value") else None
                               for dim in value.type.tensor_type.shape.dim)
              for value in infos if value.type.tensor_type.HasField("shape")}
    for node in model.graph.node:
        if node.op_type == "Constant":
            tensor = next((attr.t for attr in node.attribute if attr.name == "value"), None)
            if tensor is not None:
                shapes[node.output[0]] = tuple(tensor.dims)
    scatter_updates = {node.input[2] for node in model.graph.node if node.op_type == "ScatterND"}
    narrow = {TensorProto.BOOL, TensorProto.INT8, TensorProto.UINT8, TensorProto.INT16, TensorProto.UINT16}
    nodes = []
    last_axis = helper.make_tensor("scatter_last_axis", TensorProto.INT64, [1], [-1])
    slice_end = helper.make_tensor("scatter_slice_end", TensorProto.INT64, [1], [2**63 - 1])
    model.graph.initializer.extend((last_axis, slice_end))
    for node in model.graph.node:
        if node.op_type == "ScatterND":
            data, indices, updates = node.input
            prefix, shape, index_shape, axes, suffix, target, shaped = (
                node.name + "_" + name for name in
                ("prefix", "shape", "index_shape", "axes", "suffix", "target", "updates")
            )
            nodes.extend([
                helper.make_node("Shape", [indices], [prefix], end=-1),
                helper.make_node("Shape", [data], [shape]),
                helper.make_node("Shape", [indices], [index_shape]),
                helper.make_node("Gather", [index_shape, last_axis.name], [axes]),
                helper.make_node("Slice", [shape, axes, slice_end.name], [suffix]),
                helper.make_node("Concat", [prefix, suffix], [target], axis=0),
                helper.make_node("Reshape", [updates, target], [shaped], allowzero=1),
            ])
            node.input[2] = shaped
        if node.op_type == "Reshape" and node.output[0] in scatter_updates:
            source = shapes.get(node.input[0])
            target = shapes.get(node.input[1])
            if source is not None and target and target[0] is not None and len(source) < target[0]:
                node = helper.make_node("Expand", node.input, node.output, name=node.name)
        dtype = types.get(node.input[1]) if node.op_type == "Where" else None
        if dtype in narrow:
            a, b, out = (node.name + suffix for suffix in ("_wide_a", "_wide_b", "_wide_out"))
            nodes.extend([
                helper.make_node("Cast", [node.input[1]], [a], to=TensorProto.INT64),
                helper.make_node("Cast", [node.input[2]], [b], to=TensorProto.INT64),
                helper.make_node("Where", [node.input[0], a, b], [out]),
                helper.make_node("Cast", [out], node.output, to=dtype),
            ])
        else:
            nodes.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    model.graph.ClearField("value_info")
    return model


class Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame, action, state=None):
        pixels, carry = self.model._counter_step(frame, action, state, observed=False)
        phase = self.model.pause_phase(pixels, carry, torch.zeros_like(action, dtype=torch.bool))
        return pixels, carry, phase


def export_graph(model, frame, state, destination):
    inputs = (frame, torch.zeros(1, dtype=torch.int64))
    names = ["frame", "action"]
    if state is not None:
        inputs += (state,)
        names.append("state")
    buffer = io.BytesIO()
    step = Step(model)
    with torch.inference_mode():
        outputs = step(*inputs)
        torch.onnx.export(step, inputs, buffer, opset_version=OPSET, dynamo=False,
                          input_names=names, output_names=["pixels", "carry", "phase"])
    graph = runtime_graph(onnx.load_model_from_string(buffer.getvalue()))
    for value, tensor in zip(graph.graph.output, outputs):
        shape = value.type.tensor_type.shape
        shape.ClearField("dim")
        for size in tensor.shape:
            shape.dim.add(dim_value=size)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    for name, symbolic in (("isin", membership), ("logaddexp", logaddexp), ("expm1", expm1),
                           ("flatten", flatten), ("scatter_reduce", scatter_reduce)):
        torch.onnx.register_custom_op_symbolic(f"aten::{name}", symbolic, OPSET)
    models = args.destination / "models"
    models.mkdir(parents=True, exist_ok=True)
    (args.destination / "v1").mkdir(exist_ok=True)
    (args.destination / "v3").mkdir(exist_ok=True)
    for path, name in (("", "index.html"), ("/v1", "v1/index.html")):
        page = version_page(path, browser=True)
        page = page.replace('<nav aria-label="Game version">', '<nav aria-label="Game version"><a href="/v3">V3</a>')
        (args.destination / name).write_text(page, encoding="utf-8")
    (args.destination / "health").write_text("ok\n", encoding="utf-8")
    web = Path(__file__).with_name("web")
    for name in ("local-game.mjs", "native-render.mjs", "color-render.mjs", "v3-intro.mjs"):
        copyfile(web / name, args.destination / name)
    color_page = (web / "color.html").read_text(encoding="utf-8")
    _, marker, controls = version_page("", browser=True).partition("<script>")
    if not marker:
        raise ValueError("missing shared browser controls")
    (args.destination / "v3/index.html").write_text(color_page + marker + controls, encoding="utf-8")
    copyfile(RELEASE.with_name("functional-3") / "color.json", models / "v3-color.json")
    versions = {
        "v1": NeuralSession(RELEASE / "model.pt", RELEASE / "first-piece.npz"),
        "v2": FunctionalTwoSession(RELEASE.with_name("functional-2")),
    }
    for version, game in versions.items():
        metadata = json.dumps(release_metadata(game), separators=(",", ":"))
        (models / f"{version}.json").write_text(metadata + "\n", encoding="utf-8")
        export_graph(game.model, game.frame, None, models / f"{version}-initial.onnx")
        if version == "v2":
            game.step("start")
        game.step("noop")
        export_graph(game.model, game.frame, game.state, models / f"{version}-recurrent.onnx")


if __name__ == "__main__":
    main()
