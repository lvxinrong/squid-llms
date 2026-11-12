"""I2 环境探针：验证当前设备是否支持 FP16 / BF16、autocast、SDPA 与 torch.compile。

用途：在正式训练加速实验前区分环境能力与性能结果，尽早暴露编译器、依赖或设备缺口。
输出：data/reports/i2-environment.json，包含平台、PyTorch 版本和各探针真实结果 / 错误。
边界：SDPA 可用不等于使用 FlashAttention；compile 探针成功不代表模型训练一定提速，也不产生性能基线。
"""

import json
import platform
from pathlib import Path

import torch
import torch.nn.functional as F

from squid.device import get_device


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data/reports/i2-environment.json"

"""
判断某项功能在当前环境里能否运行
高阶函数（Higher-order Function）接收函数、返回函数，或两者都有的函数，就称为高阶函数。
*args：接收位置参数，只传值，不写参数名；
**kwargs：接收关键字参数，写成 key=value。
args 和 kwargs 本身只是约定俗成的变量名，真正让参数收集生效的是前面的 * 和 **。
"""
def run_probe(function):
    try:
        result = function()
        return {
            "supported": True,
            "result": str(result),
        }
    except Exception as exc:
        return {
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main():
    device = get_device()

    def dtype_probe(dtype):
        x = torch.ones(
            16,
            16,
            device=device,
            dtype=dtype,
        )
        result = (x @ x).sum().item()
        return f"matmul sum={result}"

    def autocast_probe(dtype):
        x = torch.randn(
            16,
            16,
            device=device,
            dtype=torch.float32,
        )
        # torch.autocast 是在指定作用域里自动为部分算子选择计算精度；device_type 和 dtype 只是它的配置参数。
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
        ):
            result = x @ x

        assert result.dtype == dtype
        return f"output dtype={result.dtype}"

    def sdpa_probe():
        q = torch.randn(
            1,
            2,
            8,
            16,
            device=device,
        )
        """
        SDPA 是 PyTorch 提供的统一注意力入口，具体使用哪种底层实现，由 PyTorch 根据当前环境和输入条件决定。
        SDPA 则只要求你提供 Q/K/V 和必要配置：
        至于内部怎样计算，由 PyTorch 根据条件选择，例如：
            PyTorch 版本；
            CPU、CUDA 或 MPS；
            dtype 是 FP32、FP16 还是 BF16；
            输入形状；
            是否有 mask；
            当前平台支持哪些优化后端。
            可能使用普通 math 路径，也可能使用 memory-efficient、FlashAttention 等优化实现。不是每个平台都会使用 FlashAttention。
            
        SDPA 是把同一件数学任务交给 PyTorch，由它根据环境选择底层实现。FlashAttention 只是 SDPA 可能使用的后端之一，二者不是同一个概念。
        
        我们的 model_v3：手写 eager attention
        官方 eager：逐步展开 attention
        官方 SDPA：内部调用 F.scaled_dot_product_attention
        """

        result = F.scaled_dot_product_attention(q, q, q)
        return f"output shape={tuple(result.shape)}"

    def compile_probe():
        def function(x):
            return x * 2 + 1
        """
        torch.compile() 返回的是一个“经过编译优化的新可调用对象”，我们把它命名为 compiled。
        torch.compile() 把 PyTorch 中逐步执行的代码，尝试转换成更适合优化的计算图，再由编译后端生成更高效的执行路径。
        torch.compile(function) 是先把函数包装成可编译优化的版本，真正传入输入时才执行；它优化的是计算执行方式，不是修改模型的数学公式或权重。

        个人理解，类似java中的JIT（Just-In-Time，即时编译），以及 Java HotSpot JVM 里的热点代码、分层编译、C1/C2 编译器和原生机器码。
        在运行的时候将热点代码对应的字节码编译并优化成当前CPU可直接执行的原生机器码。
        Java JIT 会自动根据热点程度做运行时优化，而 torch.compile() 是我们显式选择要编译的函数或模型。
        Java JIT 优化的是热点字节码到机器码的执行；torch.compile 优化的是张量计算图到设备内核的执行。两者都不是改变程序的数学逻辑，而是在保持语义的前提下，让同一段计算跑得更好。
        """

        compiled = torch.compile(function)
        x = torch.tensor([1.0, 2.0], device=device)
        result = compiled(x)
        return f"compiled result={result.tolist()}"

    report = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "probes": {
            "float16_matmul": run_probe(
                # lambda 是 Python 中创建匿名函数的关键字。匿名就是：不需要用 def 给它单独起一个函数名。
                # 基本格式: lambda 参数: 返回值表达式
                # lambda x: x * 2 等价于 def double(x): return x * 2
                # 所以这里 lambda 的核心作用是：把一次函数调用包装成一个稍后才执行的无参数函数。
                # run_probe(dtype_probe(torch.float16)) 如果这么写的话，会先执行 dtype_probe(torch.float16) -> 得到结果 -> 再调用run_probe
                # 这样，如果 dtype_probe 出错，异常发生在进入 run_probe() 之前，里面的 try/except 就接不住。
                # 这里“延迟”只是相对于直接调用而言。到了 run_probe() 内部，它仍然立即执行，没有额外的等待。
                lambda: dtype_probe(torch.float16)
            ),
            "bfloat16_matmul": run_probe(
                lambda: dtype_probe(torch.bfloat16)
            ),
            "float16_autocast": run_probe(
                lambda: autocast_probe(torch.float16)
            ),
            "bfloat16_autocast": run_probe(
                lambda: autocast_probe(torch.bfloat16)
            ),
            # 因为 sdpa_probe和compile_probe 是没有入参，直接传函数对象即可
            # run_probe(sdpa_probe)   # 传入函数对象，还没执行
            # run_probe(sdpa_probe()) # 先执行函数，再把结果传进去
            # 无参数函数直接传函数名；需要参数的函数，可以用 lambda 或 partial 把参数提前绑定好，再交给 run_probe() 统一执行。
            "sdpa": run_probe(sdpa_probe),
            "torch_compile": run_probe(compile_probe),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n环境报告已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
