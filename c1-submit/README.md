# C1 PTX-to-AEC 标量编译器实现说明

## 1. 项目目标

本项目实现 Track-C C1 赛题要求的 PTX-to-AEC 编译器。

编译器输入为 NVIDIA PTX ISA 9.3 的受限标量子集，输出为符合题目格式要求的 AEC 128-bit 定长机器码文件 `.aecbin`。实现内容包括：

1. PTX 词法与语法解析；
2. 内部 IR 和基本块构建；
3. PTX 指令到 AEC 指令的 lowering；
4. 标量优化与内存访问优化；
5. 寄存器分配与 spill 处理；
6. 指令调度；
7. AEC 二进制编码；
8. 编译报告输出；
9. AEC 反汇编。

---

## 2. 目录结构与代码入口

```text
c1-submit/
├── compiler/
│   ├── aec-cc
│   └── src/
│       ├── compiler.py
│       ├── ptx_lexer.py
│       ├── ptx_parser.py
│       ├── ir.py
│       ├── optimizer.py
│       ├── instruction_lowering.py
│       ├── memory_optimizer.py
│       ├── instruction_scheduler.py
│       ├── register_allocator.py
│       ├── binary_encoder.py
│       └── disassembler.py
├── disassembler/
│   ├── aec-objdump
│   └── src/
└── README.md
```

主要入口：

- `compiler/aec-cc`：唯一编译器入口；
- `compiler/src/compiler.py`：编译流程总控；
- `disassembler/aec-objdump`：AEC 二进制反汇编入口。

---

## 3. 运行环境

- Linux x86-64；
- Python 3.13.5；
- 仅使用 Python 标准库；
- 无需第三方 Python 依赖；
- 无需额外构建步骤。

入口文件使用 shebang，可直接执行：

```bash
chmod +x compiler/aec-cc disassembler/aec-objdump
```

---

## 4. 编译器总体流程

编译主流程位于：

```text
compiler/src/compiler.py
```

整体流程如下：

```text
PTX 源文件
  -> PTX 词法分析与语法解析
  -> PTX IR 与基本块构建
  -> PTX 层优化
  -> PTX-to-AEC lowering
  -> 冗余 global load 消除
  -> 指令调度
  -> 寄存器分配与 spill 插入
  -> AEC 二进制编码
  -> 输出 .aecbin
```

核心模块职责如下：

| 模块 | 主要职责 |
|---|---|
| `ptx_lexer.py` | PTX token 化 |
| `ptx_parser.py` | PTX 语法解析、kernel/参数/寄存器/指令识别 |
| `ir.py` | PTX IR、基本块和 AEC 指令数据结构 |
| `optimizer.py` | 常量优化、DCE、CSE、LICM、地址优化、FMA 融合 |
| `instruction_lowering.py` | PTX-to-AEC lowering、ABI 与地址处理 |
| `memory_optimizer.py` | 冗余 global load 消除 |
| `instruction_scheduler.py` | basic-block 内依赖感知调度 |
| `register_allocator.py` | 物理寄存器分配、寄存器对、spill |
| `binary_encoder.py` | AEC 128-bit 指令编码 |
| `disassembler.py` | AEC 指令解码和文本输出 |

---

## 5. `-O2` 完整优化配置

`-O2` 启用：

- 常量传播；
- 常量折叠；
- 代数化简；
- copy propagation；
- 全局和局部死代码删除；
- 局部公共子表达式消除；
- 保守的循环不变量外提；
- 仿射地址强度削弱；
- 冗余 global load 消除；
- FP32 `mul + add` 融合为 AEC `FMA`；
- 安全的 low32 地址链优化；
- 基于依赖关系的 list scheduling；
- 寄存器复用与 spill 处理。

---

## 6. T1：基础指令 Lowering

主要实现文件：

```text
compiler/src/instruction_lowering.py
```

### 6.1 Kernel 参数 ABI

编译器按照参数声明顺序计算 `.pmem` offset，并执行自然对齐：

- 32-bit 参数：4-byte size，4-byte alignment；
- 64-bit 参数：8-byte size，8-byte alignment；
- 参数块最终向 8 bytes 对齐。

例如：

```ptx
ld.param.u32 %r1, [param_n];
```

lower 为：

```text
LOADI offset_register, param_offset
LD.pmem.u32 destination, [offset_register]
```

`.u64` 和 `.b64` 参数通过两次 32-bit `.pmem` load 读取低 32 位和高 32 位。

### 6.2 Special register

支持：

- `%tid.x/y/z`
- `%ntid.x/y/z`
- `%ctaid.x/y/z`
- `%nctaid.x/y/z`
- `%laneid`

这些寄存器通过 AEC `CPY` 与 special-register selector 实现。

### 6.3 基础算术、FP32 与位运算

支持的 lowering 包括：

- `add.u32`
- `sub.u32`
- `mul.lo.u32`
- `mad.lo.u32`
- `and.b32`
- `or.b32`
- `xor.b32`
- `shl.b32`
- `shr.u32`
- `add.f32`
- `sub.f32`
- `mul.f32`
- `mad.f32`
- `fma.rn.f32`

### 6.4 比较与分支

支持：

- `setp.eq/ne/lt/le/gt/ge.u32`
- `bra LABEL`
- `@%p bra LABEL`
- `@!%p bra LABEL`

分支目标在 lowering 后统一解析为 AEC instruction PC。

### 6.5 Global memory

支持：

- `ld.global.f32/u32/b32`
- `st.global.f32/u32/b32`

global memory 地址在 PTX 中使用 64-bit 虚拟寄存器表示，AEC `LD/ST` 根据 C1 abstract-address 规则使用地址低 32 位。

### 6.6 Kernel 返回

```ptx
ret;
```

lower 为：

```text
HALT
```

---

## 7. 64-bit 地址处理

PTX `.u64` 和 `.b64` 虚拟寄存器映射为 AEC GPR pair：

```text
low 32 bits  -> Rk
high 32 bits -> Rk+1
```

普通 `mul.wide.u32` lowering：

```text
MUL.u32 low, src1, src2
LOADI high, 0
```

普通 `add.u64` lowering：

```text
ADD.u32 dst_low, src1_low, src2_low
LOADI dst_high, 0
```

### 7.1 low32 地址链优化

在 `-O2` 下，编译器会对 `mul.wide.u32` 和 `add.u64` 结果执行完整用途分析。

只有当编译器证明某个 64-bit 中间值：

1. 仅沿地址计算链继续传播；
2. 最终只用于 `ld.global` 或 `st.global`；
3. 未参与普通 64-bit 数据计算；

才会省略高 32 位的 `LOADI 0`。

这样可以减少地址计算指令，同时避免错误优化普通 64-bit 数据。

---

## 8. T2：标量优化

主要实现文件：

```text
compiler/src/optimizer.py
```

### 8.1 常量传播与常量折叠

实现内容包括：

- immediate 传播；
- 32-bit 整数常量计算；
- 常量 move 消除；
- 可确定结果直接替换为 immediate。

### 8.2 代数化简

例如：

```text
x + 0 -> x
x - 0 -> x
x * 1 -> x
x * 0 -> 0
```

优化保持 32-bit modulo 语义。

### 8.3 公共子表达式消除

在 basic block 内识别相同 opcode、type 和 operand 的重复表达式并复用结果。

当相关寄存器被重新定义时，对应表达式会失效。

### 8.4 Copy propagation

传播寄存器复制关系并删除：

- 可替代 copy；
- self-copy；
- 已被完全传播的临时值。

### 8.5 死代码删除

采用 function 范围和 basic-block 范围两层处理。

具有 memory side effect、branch 或 control effect 的指令不会被删除。

### 8.6 循环不变量外提

仅对结构明确、具有唯一 preheader 的简单自然循环执行。

候选指令必须满足：

- 无 memory side effect；
- 无 predicate 或 control side effect；
- 输入在循环内不变化；
- 移动后不破坏数据依赖。

---

## 9. T3：内存访问与地址优化

主要实现文件：

```text
compiler/src/memory_optimizer.py
compiler/src/optimizer.py
compiler/src/instruction_lowering.py
```

### 9.1 冗余 global load 消除

优化器跟踪：

- load 地址；
- load type；
- 已加载结果；
- 地址寄存器重定义；
- store 对 memory state 的影响；
- branch 和 block boundary。

只有在可证明安全时才复用 global load。遇到可能改变 memory state 或控制流的情况时会保守清除复用信息。

### 9.2 仿射地址强度削弱

针对循环内重复地址计算，将重复乘法计算转换为指针递增形式。

优化过程检查：

- 循环结构；
- preheader；
- induction variable；
- stride；
- 地址用途；
- 寄存器冲突。

### 9.3 地址链高位写入消除

结合 low32 地址用途分析，删除已证明无用的高 32 位写入，降低地址计算开销。

---

## 10. T4：寄存器分配与指令调度

### 10.1 寄存器分配

文件：

```text
compiler/src/register_allocator.py
```

实现：

- 虚拟 GPR 到 AEC 物理 GPR 映射；
- 64-bit register pair 约束；
- register index 检查；
- predicate index 检查；
- opcode/type 合法性检查；
- branch target 检查；
- `LOADI64` register pair 越界检查。

### 10.2 Spill 处理

当寄存器压力超过可用物理寄存器范围时：

1. 为 spilled virtual register 分配 local-memory slot；
2. 在使用前插入 `LD.lmem`；
3. 在定义后插入 `ST.lmem`；
4. 使用临时物理寄存器执行原指令；
5. 在编译报告中统计 spill load/store。

### 10.3 指令调度

文件：

```text
compiler/src/instruction_scheduler.py
```

采用 basic-block 内 list scheduling。

依赖分析覆盖：

- GPR RAW；
- GPR WAR；
- GPR WAW；
- predicate 依赖；
- memory 顺序依赖；
- control instruction 边界。

调度器优先尽早安排 load，并在 load 与消费者之间插入独立计算，同时保持所有依赖关系正确。

---

## 11. T5：FP32 Scalar GEMM

T5 综合涉及：

- 二维 thread 索引；
- 矩阵边界判断；
- K 维循环；
- A/B 地址计算；
- global FP32 load；
- FP32 multiply-add；
- C 地址计算与 store。

本实现针对 T5 使用通用分析和优化，不依赖固定变量名、寄存器名、label 名或矩阵大小。

主要优化：

1. 常量传播与代数化简；
2. 循环不变量外提；
3. 仿射地址强度削弱；
4. copy propagation；
5. FP32 `mul + add` 融合为 AEC `FMA`；
6. 冗余 global load 消除；
7. 地址链高位无效写入消除；
8. load/compute interleaving；
9. 物理寄存器复用。

---

## 12. AEC 二进制编码

文件：

```text
compiler/src/binary_encoder.py
```

每条 AEC 指令编码为 128 bits：

```text
bits [127:112]  Opcode
bits [111:96]   Pred/Ctrl
bits [95:80]    Dest
bits [79:64]    Src1
bits [63:32]    Src2/Imm32
bits [31:0]     ImmExt
```

文件存储顺序：

```text
w0, w1, w2, w3
```

每个 word 为 little-endian 32-bit。

输出满足：

- 无 header；
- 无 relocation；
- 无 symbol table；
- entry PC 为 0；
- 所有 label 和 branch target 在编译期解析；
- 文件大小为 16 bytes 的整数倍。

---

## 13. 编译报告

正式评测命令包含：

```bash
--report compile_report.json
```

报告内容包括：

- 编译状态；
- 输入和输出路径；
- 优化等级；
- PTX 指令数；
- AEC 指令数；
- basic block 数；
- 虚拟寄存器数；
- 物理寄存器数；
- predicate 数；
- spill load/store 数；
- 优化 pass 统计；
- scheduler 类型；
- warning 或 error 信息。

---

## 14. 使用方法

### 编译

```bash
./compiler/aec-cc input.ptx -O2 -o output.aecbin \
  --report compile_report.json
```

### 反汇编

```bash
./disassembler/aec-objdump output.aecbin
```

## AI 辅助开发声明

本项目开发过程中使用了大语言模型辅助代码分析、调试与文档整理。所有相关代码均由参赛队成员理解、检查、修改并完成测试验证。
