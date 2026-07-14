from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class Register:
    kind: str  
    num: int  
    size: int  

    def __hash__(self):
        return hash((self.kind, self.num))

    def __eq__(self, other):
        return isinstance(other, Register) and self.kind == other.kind and self.num == other.num

    def __repr__(self):
        return f"%{self.kind}{self.num}"


@dataclass
class Immediate:
    value: Any
    dtype: str

    def __repr__(self):
        return f"{self.value}"


@dataclass
class MemoryOperand:
    base: Optional[Register]
    offset: Optional[Any]
    space: str

    def __repr__(self):
        if self.offset is None:
            return f"[{self.base}]"
        return f"[{self.base} + {self.offset}]"


@dataclass(frozen=True)
class KernelParam:
    name: str
    dtype: str

    def __repr__(self):
        return f".param .{self.dtype} {self.name}"


@dataclass
class PTXInstruction:
    opcode: str
    dtype: str
    dest: Optional[Register]
    srcs: List[Any]
    label: Optional[str] = None
    pred: Optional[str] = None

    def __repr__(self):
        parts = []
        if self.pred:
            parts.append(f"@{self.pred}")
        parts.append(f"{self.opcode}.{self.dtype}" if self.dtype else self.opcode)
        if self.dest:
            parts.append(str(self.dest))
        for src in self.srcs:
            parts.append(str(src))
        return " ".join(parts)


@dataclass
class BasicBlock:
    name: str
    instructions: List[PTXInstruction] = field(default_factory=list)
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"BasicBlock({self.name}, {len(self.instructions)} instructions)"


@dataclass
class Function:
    name: str
    params: List[KernelParam] = field(default_factory=list)
    registers: Dict[str, List[Register]] = field(default_factory=dict)
    blocks: Dict[str, BasicBlock] = field(default_factory=dict)
    entry_block: str = ""

    def __repr__(self):
        return f"Function({self.name}, {len(self.blocks)} blocks)"


@dataclass
class PTXProgram:
    version: str
    target: str
    address_size: int
    functions: List[Function] = field(default_factory=list)

    def __repr__(self):
        return f"PTXProgram({len(self.functions)} functions)"


@dataclass
class AECInstruction:
    opcode: int
    dtype: int
    pred: int
    control: int
    dst: int
    src1: int
    src2: int
    src3: int
    immediate: int = 0
    modifier: int = 0
    original: Optional[PTXInstruction] = None

    def __repr__(self):
        op_names = {
            0x0001: "ADD", 0x0002: "SUB", 0x0003: "MUL", 0x0004: "MAD",
            0x0005: "FMA", 0x0006: "DIV", 0x0007: "NEG", 0x0008: "ABS",
            0x0009: "MIN", 0x000a: "MAX", 0x0010: "AND", 0x0011: "OR",
            0x0012: "XOR", 0x0013: "NOT", 0x0014: "SHL", 0x0015: "SHR",
            0x0016: "BFX", 0x0017: "BINS", 0x0018: "POPC", 0x0019: "FLO",
            0x0020: "CMP", 0x0021: "CMPP", 0x0022: "SEL", 0x0023: "PICK",
            0x0030: "LD", 0x0031: "ST", 0x0032: "LDC", 0x0033: "ATOM",
            0x0040: "BR", 0x0041: "BRX", 0x0042: "JMP", 0x0043: "CALL",
            0x0044: "RET", 0x0045: "HALT", 0x0046: "SSYNC", 0x0047: "SYNC_CT",
            0x0048: "SYNC_WG", 0x0049: "MBAR", 0x0050: "LOADI", 0x0051: "CPY",
            0x0052: "LOADI64", 0x0053: "CVTFF", 0x0054: "CVTFI", 0x0055: "CVTIF",
            0x0056: "CVTII", 0x0057: "SHUF", 0x0058: "VOTE", 0x0059: "MTCH",
            0x0060: "TMUL", 0x0061: "TMUL_S", 0x0062: "TLDA", 0x0063: "TSTA",
            0x0064: "TMOV", 0x0065: "TDUP", 0x0070: "RCP", 0x0071: "RSQ",
            0x0072: "SIN", 0x0073: "COS", 0x0074: "EXP", 0x0075: "LOG",
            0x0076: "SQRT", 0x0080: "RDTSC", 0x0081: "RDPMC"
        }
        type_names = {
            0: "f32", 1: "f64", 2: "f16", 3: "bf16", 4: "f8e4m3", 5: "f8e5m2",
            6: "f4e2m1", 7: "s32", 8: "u32", 9: "s8", 10: "u8", 11: "s4",
            12: "u4", 13: "b32", 14: "b64"
        }
        op_name = op_names.get(self.opcode, f"0x{self.opcode:04x}")
        type_name = type_names.get(self.dtype, f"0x{self.dtype:x}")
        parts = [f"{op_name}.{type_name}"]
        if self.pred != 15:
            parts.append(f"@P{self.pred}")
        if self.dst != 0:
            parts.append(f"R{self.dst}")
        if self.src1 != 0:
            parts.append(f", R{self.src1}")
        if self.src2 != 0:
            parts.append(f", R{self.src2}")
        if self.src3 != 0:
            parts.append(f", R{self.src3}")
        if self.immediate != 0:
            parts.append(f", 0x{self.immediate:x}")
        return "".join(parts)
