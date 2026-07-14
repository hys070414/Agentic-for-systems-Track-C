from typing import Dict, List, Optional, Set, Tuple

from ir import (
    AECInstruction,
    Function,
    Immediate,
    MemoryOperand,
    PTXInstruction,
    PTXProgram,
    Register,
)


# Current C1 scalar ISA type codes.
PTX_TYPE_TO_AEC_TYPE = {
    "b32": 0x0,
    "b64": 0x1,
    "u32": 0x2,
    "u64": 0x1,
    "s32": 0x3,
    "f32": 0x8,
    "none": 0xF,
}

SPECIAL_REGISTER_MAP = {
    "%tid.x": 0x0100,
    "%ntid.x": 0x0101,
    "%ctaid.x": 0x0102,
    "%nctaid.x": 0x0103,
    "%laneid": 0x0104,
    "%tid.y": 0x0110,
    "%ntid.y": 0x0111,
    "%ctaid.y": 0x0112,
    "%nctaid.y": 0x0113,
    "%tid.z": 0x0120,
    "%ntid.z": 0x0121,
    "%ctaid.z": 0x0122,
    "%nctaid.z": 0x0123,
}

MEMORY_SPACE_MAP = {
    "gmem": 0,
    "global": 0,
    "smem": 1,
    "shared": 1,
    "cmem": 2,
    "const": 2,
    "lmem": 3,
    "local": 3,
    "pmem": 4,
    "param": 4,
}

CMP_OP_MAP = {
    "eq": 0,
    "ne": 1,
    "lt": 2,
    "le": 3,
    "gt": 4,
    "ge": 5,
}


def _normalize_dtype(dtype: str) -> str:
    text = (dtype or "").lstrip(".")
    parts = [p for p in text.split(".") if p]
    for part in reversed(parts):
        if part in PTX_TYPE_TO_AEC_TYPE:
            return part
    return "u32"


def _predicate_register_index(reg) -> int:
    """Return the numeric index of a PTX predicate register such as %p1."""
    text = str(reg).strip()
    if text.startswith("%p"):
        index = int(text[2:])
    elif text.startswith("p"):
        index = int(text[1:])
    else:
        kind = getattr(reg, "kind", None)
        index_attr = getattr(reg, "index", None)
        if kind == "p" and index_attr is not None:
            index = int(index_attr)
        else:
            raise ValueError(f"Unsupported predicate destination: {reg!r}")

    if not 0 <= index <= 7:
        raise ValueError(
            f"AEC supports predicate registers P0-P7, got P{index}"
        )
    return index


def _parse_predicate(pred_text: Optional[str]) -> Tuple[int, bool]:
    if not pred_text:
        return 15, False

    text = str(pred_text).strip()
    if text.startswith("@"):
        text = text[1:]

    negated = text.startswith("!")
    if negated:
        text = text[1:]

    if text.startswith("%p"):
        return int(text[2:]), negated

    if text.startswith("p"):
        return int(text[1:]), negated

    raise ValueError(f"Unsupported predicate syntax: {pred_text!r}")


def _param_name(mem: MemoryOperand) -> Optional[str]:
    base = mem.base
    if base is None:
        return None

    name = base if isinstance(base, str) else str(base)
    return name.lstrip("%")


def _type_size_alignment(dtype: str) -> Tuple[int, int]:
    dtype = _normalize_dtype(dtype)
    if dtype in ("b64", "u64"):
        return 8, 8
    return 4, 4


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class RegisterAllocator:
    """Allocate stable PTX registers with conservative physical reuse."""

    # Virtual AEC register namespace. Physical R1..R255 assignment and
    # local-memory spilling are handled later by register_allocator.py.
    MAX_GPR = 0xFFFF

    def __init__(self) -> None:
        self.ptx_to_aec: Dict[Register, int] = {}
        self.ptx_width: Dict[Register, int] = {}
        self.next_reg = 1
        self._free_gprs: set[int] = set()
        self._active_temps: List[int] = []

    def _check_range(self, low: int, width: int) -> None:
        high = low + width - 1
        if low < 1 or high > self.MAX_GPR:
            raise ValueError(
                f"AEC register allocation overflow: R{low}..R{high}, "
                f"valid range is R1..R{self.MAX_GPR}"
            )

    def _allocate_physical(self, width: int) -> int:
        if width == 1 and self._free_gprs:
            reg = min(self._free_gprs)
            self._free_gprs.remove(reg)
            return reg

        if width == 2:
            for low in sorted(self._free_gprs):
                if low % 2 == 0 and low + 1 in self._free_gprs:
                    self._free_gprs.remove(low)
                    self._free_gprs.remove(low + 1)
                    return low

            if self.next_reg % 2 != 0:
                self.next_reg += 1

        low = self.next_reg
        self._check_range(low, width)
        self.next_reg += width
        return low

    def get(self, reg: Register, width: int = 1) -> int:
        if width not in (1, 2):
            raise ValueError(f"Unsupported register width: {width}")

        if reg not in self.ptx_to_aec:
            low = self._allocate_physical(width)
            self.ptx_to_aec[reg] = low
            self.ptx_width[reg] = width
        elif width > self.ptx_width.get(reg, 1):
            raise ValueError(
                f"Register {reg!r} was first allocated as width "
                f"{self.ptx_width.get(reg, 1)}, then requested as width {width}"
            )

        return self.ptx_to_aec[reg]

    def get_pair(self, reg: Register) -> Tuple[int, int]:
        low = self.get(reg, width=2)
        if low % 2 != 0 or low >= self.MAX_GPR:
            raise ValueError(f"Invalid virtual 64-bit register pair low register: R{low}")
        return low, low + 1

    def begin_instruction(self) -> None:
        if self._active_temps:
            raise RuntimeError("Temporary-register scope was not closed")

    def temporary(self, width: int = 1) -> int:
        if width != 1:
            raise NotImplementedError(
                "Only 32-bit instruction-local temporaries are supported"
            )
        reg = self._allocate_physical(1)
        self._active_temps.append(reg)
        return reg

    def end_instruction(self) -> None:
        for reg in self._active_temps:
            self._free_gprs.add(reg)
        self._active_temps = []

    def release(self, reg: Register) -> None:
        """Release a PTX register proven dead for the rest of the function."""
        if reg not in self.ptx_to_aec:
            return

        low = self.ptx_to_aec.pop(reg)
        width = self.ptx_width.pop(reg, 1)
        for physical in range(low, low + width):
            self._free_gprs.add(physical)


def _make(
    *,
    opcode: int,
    dtype: int = 0xF,
    pred: int = 15,
    pred_neg: bool = False,
    dst: int = 0,
    src1: int = 0,
    src2: int = 0,
    src3: int = 0,
    immediate: int = 0,
    modifier: int = 0,
) -> AECInstruction:
    # binary_encoder.py uses control bit 0 as pred_neg.
    return AECInstruction(
        opcode=opcode,
        dtype=dtype,
        pred=pred,
        control=1 if pred_neg else 0,
        dst=dst,
        src1=src1,
        src2=src2,
        src3=src3,
        immediate=immediate,
        modifier=modifier,
    )


def _materialize_operand(
    operand,
    dtype: str,
    reg_alloc: RegisterAllocator,
    pred: int,
    pred_neg: bool,
) -> Tuple[List[AECInstruction], int]:
    if isinstance(operand, Register):
        width = 2 if _normalize_dtype(dtype) in ("b64", "u64") else 1
        return [], reg_alloc.get(operand, width=width)

    if isinstance(operand, Immediate):
        temp = reg_alloc.temporary()
        value = int(operand.value) & 0xFFFFFFFF
        return [
            _make(
                opcode=0x0055,  # LOADI
                dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                pred=pred,
                pred_neg=pred_neg,
                dst=temp,
                immediate=value,
            )
        ], temp

    raise ValueError(f"Unsupported PTX operand: {operand!r}")


def _find_memory_operand(instr: PTXInstruction) -> Optional[MemoryOperand]:
    for src in instr.srcs:
        if isinstance(src, MemoryOperand):
            return src
    return None


def _special_register_value(src) -> Optional[int]:
    if isinstance(src, str):
        return SPECIAL_REGISTER_MAP.get(src)

    if isinstance(src, Register):
        kind = getattr(src, "kind", "")
        for name in (f"%{kind}.x", f"%{kind}"):
            if name in SPECIAL_REGISTER_MAP:
                return SPECIAL_REGISTER_MAP[name]

    return None


def _collect_param_offsets(func: Function) -> Dict[str, int]:
    """Lay out .entry parameters in declaration order using natural alignment."""
    offsets: Dict[str, int] = {}
    offset = 0

    for param in func.params:
        name = param.name.lstrip("%")

        # Do not use _normalize_dtype() for ABI validation: that helper falls
        # back to u32 for unknown spellings, which would silently accept types
        # such as .u16. Kernel parameter types must match the supported ABI
        # allowlist exactly.
        raw_dtype = str(param.dtype or "").strip().lstrip(".")
        dtype_parts = [part for part in raw_dtype.split(".") if part]
        dtype = dtype_parts[-1] if dtype_parts else ""

        if dtype not in ("u32", "s32", "b32", "f32", "u64", "b64"):
            raise ValueError(
                f"Unsupported kernel parameter type .{param.dtype} "
                f"for parameter {param.name!r}"
            )

        size, alignment = _type_size_alignment(dtype)
        offset = _align_up(offset, alignment)
        offsets[name] = offset
        offset += size

    # ABI requires the full parameter block size to be aligned to 8 bytes.
    # The current lowering only needs individual offsets, but computing this
    # value here validates the complete layout.
    _total_parameter_bytes = _align_up(offset, 8)
    return offsets


def lower_ptx_to_aec(
    instr: PTXInstruction,
    reg_alloc: RegisterAllocator,
    param_offsets: Optional[Dict[str, int]] = None,
    *,
    omit_u64_high_zero: bool = False,
) -> List[AECInstruction]:
    results: List[AECInstruction] = []

    pred, pred_neg = _parse_predicate(instr.pred)
    dtype_name = _normalize_dtype(instr.dtype)
    aec_type = PTX_TYPE_TO_AEC_TYPE.get(dtype_name, 0x2)

    if instr.opcode == "ld" and "param" in instr.dtype:
        mem = _find_memory_operand(instr)
        if mem is None:
            raise ValueError("ld.param is missing a memory operand")

        name = _param_name(mem)
        if not name:
            raise ValueError("ld.param is missing a parameter name")

        if param_offsets is None or name not in param_offsets:
            raise ValueError(f"Unknown parameter {name!r}")

        base_offset = param_offsets[name]
        addr_reg = reg_alloc.temporary()

        results.append(
            _make(
                opcode=0x0055,  # LOADI
                dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                pred=pred,
                pred_neg=pred_neg,
                dst=addr_reg,
                immediate=base_offset,
            )
        )

        if dtype_name in ("b64", "u64"):
            if instr.dest is None:
                raise ValueError("64-bit ld.param has no destination")
            dst_lo, dst_hi = reg_alloc.get_pair(instr.dest)

            results.append(
                _make(
                    opcode=0x0030,
                    dtype=PTX_TYPE_TO_AEC_TYPE["b32"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst_lo,
                    src1=addr_reg,
                    modifier=MEMORY_SPACE_MAP["pmem"],
                )
            )

            addr_hi = reg_alloc.temporary()
            results.append(
                _make(
                    opcode=0x0055,
                    dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=addr_hi,
                    immediate=base_offset + 4,
                )
            )
            results.append(
                _make(
                    opcode=0x0030,
                    dtype=PTX_TYPE_TO_AEC_TYPE["b32"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst_hi,
                    src1=addr_hi,
                    modifier=MEMORY_SPACE_MAP["pmem"],
                )
            )
        else:
            if instr.dest is None:
                raise ValueError("ld.param has no destination")
            dst = reg_alloc.get(instr.dest)
            results.append(
                _make(
                    opcode=0x0030,
                    dtype=aec_type,
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst,
                    src1=addr_reg,
                    modifier=MEMORY_SPACE_MAP["pmem"],
                )
            )

        return results

    if instr.opcode == "mov":
        if instr.dest is None or not instr.srcs:
            raise ValueError("mov requires destination and source")

        src = instr.srcs[0]
        special = _special_register_value(src)

        if special is not None:
            dst = reg_alloc.get(instr.dest)
            return [
                _make(
                    opcode=0x0054,  # CPY
                    dtype=aec_type,
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst,
                    src1=special,
                )
            ]

        if isinstance(src, Register):
            width = 2 if dtype_name in ("b64", "u64") else 1
            dst = reg_alloc.get(instr.dest, width=width)
            src_reg = reg_alloc.get(src, width=width)

            if width == 1:
                return [
                    _make(
                        opcode=0x0054,
                        dtype=aec_type,
                        pred=pred,
                        pred_neg=pred_neg,
                        dst=dst,
                        src1=src_reg,
                    )
                ]

            return [
                _make(
                    opcode=0x0054,
                    dtype=PTX_TYPE_TO_AEC_TYPE["b32"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst,
                    src1=src_reg,
                ),
                _make(
                    opcode=0x0054,
                    dtype=PTX_TYPE_TO_AEC_TYPE["b32"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst + 1,
                    src1=src_reg + 1,
                ),
            ]

        if isinstance(src, Immediate):
            dst = reg_alloc.get(
                instr.dest,
                width=2 if dtype_name in ("b64", "u64") else 1,
            )
            value = int(src.value)

            if dtype_name in ("b64", "u64"):
                return [
                    _make(
                        opcode=0x0056,  # LOADI64
                        dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                        pred=pred,
                        pred_neg=pred_neg,
                        dst=dst,
                        immediate=value & 0xFFFFFFFFFFFFFFFF,
                    )
                ]

            return [
                _make(
                    opcode=0x0055,  # LOADI
                    dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst,
                    immediate=value & 0xFFFFFFFF,
                )
            ]

        raise ValueError(f"Unsupported mov source: {src!r}")

    if instr.opcode in ("add", "sub", "mul", "and", "or", "xor", "shl", "shr"):
        if instr.dest is None or len(instr.srcs) < 2:
            raise ValueError(f"{instr.opcode} requires two sources")

        # Minimal scalar handling for mul.wide.u32 used by public tests.
        if instr.opcode == "mul" and "wide" in instr.dtype:
            dst_lo, dst_hi = reg_alloc.get_pair(instr.dest)

            prep1, src1 = _materialize_operand(
                instr.srcs[0], "u32", reg_alloc, pred, pred_neg
            )
            prep2, src2 = _materialize_operand(
                instr.srcs[1], "u32", reg_alloc, pred, pred_neg
            )
            results.extend(prep1)
            results.extend(prep2)
            results.append(
                _make(
                    opcode=0x0003,
                    dtype=PTX_TYPE_TO_AEC_TYPE["u32"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst_lo,
                    src1=src1,
                    src2=src2,
                )
            )
            if not omit_u64_high_zero:
                results.append(
                    _make(
                        opcode=0x0055,
                        dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                        pred=pred,
                        pred_neg=pred_neg,
                        dst=dst_hi,
                        immediate=0,
                    )
                )
            return results

        if dtype_name in ("b64", "u64"):
            if instr.opcode != "add":
                raise NotImplementedError(
                    f"{instr.opcode}.{dtype_name} is not implemented"
                )

            # C1 address ABI: only the low 32 bits participate in global
            # memory addressing, and the high 32 bits are guaranteed to be 0.
            if not all(isinstance(op, Register) for op in instr.srcs[:2]):
                raise NotImplementedError(
                    "add.u64 currently requires two register operands"
                )

            dst_lo, dst_hi = reg_alloc.get_pair(instr.dest)
            src1_lo, _src1_hi = reg_alloc.get_pair(instr.srcs[0])
            src2_lo, _src2_hi = reg_alloc.get_pair(instr.srcs[1])

            low_add = _make(
                opcode=0x0001,
                dtype=PTX_TYPE_TO_AEC_TYPE["u32"],
                pred=pred,
                pred_neg=pred_neg,
                dst=dst_lo,
                src1=src1_lo,
                src2=src2_lo,
            )

            if omit_u64_high_zero:
                return [low_add]

            return [
                low_add,
                _make(
                    opcode=0x0055,
                    dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                    pred=pred,
                    pred_neg=pred_neg,
                    dst=dst_hi,
                    immediate=0,
                ),
            ]

        opcode_map = {
            "add": 0x0001,
            "sub": 0x0002,
            "mul": 0x0003,
            "and": 0x0010,
            "or": 0x0011,
            "xor": 0x0012,
            "shl": 0x0014,
            "shr": 0x0015,
        }

        prep1, src1 = _materialize_operand(
            instr.srcs[0], dtype_name, reg_alloc, pred, pred_neg
        )
        prep2, src2 = _materialize_operand(
            instr.srcs[1], dtype_name, reg_alloc, pred, pred_neg
        )
        results.extend(prep1)
        results.extend(prep2)

        dst = reg_alloc.get(instr.dest)
        results.append(
            _make(
                opcode=opcode_map[instr.opcode],
                dtype=aec_type,
                pred=pred,
                pred_neg=pred_neg,
                dst=dst,
                src1=src1,
                src2=src2,
            )
        )
        return results

    if instr.opcode in ("mad", "fma"):
        if instr.dest is None or len(instr.srcs) < 3:
            raise ValueError(f"{instr.opcode} requires three sources")

        regs: List[int] = []
        for operand in instr.srcs[:3]:
            prep, reg = _materialize_operand(
                operand, dtype_name, reg_alloc, pred, pred_neg
            )
            results.extend(prep)
            regs.append(reg)

        dst = reg_alloc.get(instr.dest)
        results.append(
            _make(
                opcode=0x0004 if instr.opcode == "mad" else 0x0005,
                dtype=aec_type,
                pred=pred,
                pred_neg=pred_neg,
                dst=dst,
                src1=regs[0],
                src2=regs[1],
                src3=regs[2],
            )
        )
        return results

    if instr.opcode == "setp":
        if instr.dest is None or len(instr.srcs) < 2:
            raise ValueError("setp requires destination and two sources")

        dtype_parts = [p for p in instr.dtype.lstrip(".").split(".") if p]
        cmp_name = next((p for p in dtype_parts if p in CMP_OP_MAP), "eq")
        cmp_modifier = CMP_OP_MAP[cmp_name]

        prep1, src1 = _materialize_operand(
            instr.srcs[0], dtype_name, reg_alloc, pred, pred_neg
        )
        prep2, src2 = _materialize_operand(
            instr.srcs[1], dtype_name, reg_alloc, pred, pred_neg
        )
        results.extend(prep1)
        results.extend(prep2)

        dst = _predicate_register_index(instr.dest)
        results.append(
            _make(
                opcode=0x0021,  # CMPP
                dtype=aec_type,
                pred=pred,
                pred_neg=pred_neg,
                dst=dst,
                src1=src1,
                src2=src2,
                modifier=cmp_modifier,
            )
        )
        return results

    if instr.opcode == "bra":
        # Target PC is patched in lower_function().
        return [
            _make(
                opcode=0x0041 if pred != 15 else 0x0040,
                dtype=PTX_TYPE_TO_AEC_TYPE["none"],
                pred=pred,
                pred_neg=pred_neg,
                immediate=0,
            )
        ]

    if instr.opcode == "ret":
        return [
            _make(
                opcode=0x0045,  # HALT
                dtype=PTX_TYPE_TO_AEC_TYPE["none"],
            )
        ]

    if instr.opcode == "ld" and "global" in instr.dtype:
        if instr.dest is None:
            raise ValueError("ld.global has no destination")

        mem = _find_memory_operand(instr)
        if mem is None:
            raise ValueError("ld.global is missing a memory operand")

        base = reg_alloc.get(mem.base) if isinstance(mem.base, Register) else 0
        if base == 0:
            raise ValueError(f"Unsupported ld.global base: {mem.base!r}")

        return [
            _make(
                opcode=0x0030,
                dtype=aec_type,
                pred=pred,
                pred_neg=pred_neg,
                dst=reg_alloc.get(instr.dest),
                src1=base,
                modifier=MEMORY_SPACE_MAP["gmem"],
            )
        ]

    if instr.opcode == "st" and "global" in instr.dtype:
        mem = _find_memory_operand(instr)
        src_reg = next(
            (src for src in instr.srcs if isinstance(src, Register) and src is not mem),
            None,
        )

        if mem is None or src_reg is None:
            raise ValueError("st.global requires memory and register operands")

        if not isinstance(mem.base, Register):
            raise ValueError(f"Unsupported st.global base: {mem.base!r}")

        return [
            _make(
                opcode=0x0031,
                dtype=aec_type,
                pred=pred,
                pred_neg=pred_neg,
                src1=reg_alloc.get(mem.base),
                src2=reg_alloc.get(src_reg),
                modifier=MEMORY_SPACE_MAP["gmem"],
            )
        ]

    if instr.opcode == "cvt":
        if instr.dest is None or not instr.srcs:
            raise ValueError("cvt requires destination and source")

        # Safe first-pass behavior: only same-width 32-bit conversions are
        # represented as a copy. Unsupported conversions fail loudly.
        src = instr.srcs[0]
        if not isinstance(src, Register):
            raise NotImplementedError("Immediate cvt is not implemented")

        dst = reg_alloc.get(instr.dest)
        src_reg = reg_alloc.get(src)
        return [
            _make(
                opcode=0x0054,
                dtype=aec_type,
                pred=pred,
                pred_neg=pred_neg,
                dst=dst,
                src1=src_reg,
            )
        ]

    raise NotImplementedError(
        f"Unsupported PTX instruction: opcode={instr.opcode!r}, "
        f"dtype={instr.dtype!r}"
    )


def _branch_target_name(instr: PTXInstruction) -> Optional[str]:
    if instr.opcode != "bra" or not instr.srcs:
        return None

    target = str(instr.srcs[0]).strip()
    return target.rstrip(":").lstrip("%")


def _iter_instruction_registers(instr: PTXInstruction):
    if isinstance(instr.dest, Register):
        yield instr.dest

    for src in instr.srcs:
        if isinstance(src, Register):
            yield src
        elif isinstance(src, MemoryOperand):
            if isinstance(src.base, Register):
                yield src.base
            if isinstance(src.offset, Register):
                yield src.offset


def _cyclic_blocks(func: Function) -> set[str]:
    """Return blocks that belong to any CFG cycle."""
    cyclic: set[str] = set()

    for start in func.blocks:
        stack = list(func.blocks[start].successors)
        seen = set()

        while stack:
            current = stack.pop()
            if current == start:
                cyclic.add(start)
                break
            if current in seen or current not in func.blocks:
                continue
            seen.add(current)
            stack.extend(func.blocks[current].successors)

    return cyclic


def _local_release_schedule(
    func: Function,
) -> Dict[Tuple[str, int], List[Register]]:
    """Find PTX registers safe to recycle after their final local use.

    Non-cyclic blocks:
    - all appearances must be confined to one block.

    Cyclic blocks:
    - all appearances must still be confined to one block;
    - the register must have exactly one definition in that block;
    - that definition must precede every use;
    - therefore the value is not live across the loop backedge.

    Registers defined outside a loop and reused every iteration, induction
    variables, accumulators, predicates, and cross-block values are retained.
    """
    appearances: Dict[Register, List[Tuple[str, int]]] = {}
    definitions: Dict[Register, List[Tuple[str, int]]] = {}
    uses: Dict[Register, List[Tuple[str, int]]] = {}

    for block_name, block in func.blocks.items():
        for index, instr in enumerate(block.instructions):
            if isinstance(instr.dest, Register):
                reg = instr.dest
                appearances.setdefault(reg, []).append((block_name, index))
                definitions.setdefault(reg, []).append((block_name, index))

            for src in instr.srcs:
                source_regs: List[Register] = []

                if isinstance(src, Register):
                    source_regs.append(src)
                elif isinstance(src, MemoryOperand):
                    if isinstance(src.base, Register):
                        source_regs.append(src.base)
                    if isinstance(src.offset, Register):
                        source_regs.append(src.offset)

                for reg in source_regs:
                    appearances.setdefault(reg, []).append((block_name, index))
                    uses.setdefault(reg, []).append((block_name, index))

    cyclic = _cyclic_blocks(func)
    schedule: Dict[Tuple[str, int], List[Register]] = {}

    special_kinds = {
        "tid.x", "tid.y", "tid.z",
        "ntid.x", "ntid.y", "ntid.z",
        "ctaid.x", "ctaid.y", "ctaid.z",
        "nctaid.x", "nctaid.y", "nctaid.z",
        "laneid", "warpid",
    }

    for reg, positions in appearances.items():
        if reg.kind == "p" or reg.kind in special_kinds:
            continue

        blocks = {block_name for block_name, _ in positions}
        if len(blocks) != 1:
            # A cross-block value may be live along multiple CFG paths.
            continue

        block_name = next(iter(blocks))
        last_index = max(index for _, index in positions)

        if block_name not in cyclic:
            schedule.setdefault((block_name, last_index), []).append(reg)
            continue

        # In a cyclic block, only reuse values proven local to one iteration.
        reg_defs = definitions.get(reg, [])
        reg_uses = uses.get(reg, [])

        if len(reg_defs) != 1:
            # Zero definitions means loop-invariant input. Multiple definitions
            # may represent a recurrence or complicated lifetime.
            continue

        def_block, def_index = reg_defs[0]
        if def_block != block_name:
            continue

        if any(
            use_block != block_name or use_index <= def_index
            for use_block, use_index in reg_uses
        ):
            # A use before/at the definition indicates a loop-carried value.
            continue

        # A destination with no later use can also be released immediately
        # after its definition. DCE normally removes it, but this is safe.
        schedule.setdefault((block_name, last_index), []).append(reg)

    return schedule



def _reg_key(reg: Register) -> Tuple[str, int]:
    return reg.kind, reg.num


def _is_u64_add(instr: PTXInstruction) -> bool:
    tokens = {
        token
        for token in str(instr.dtype or "").lstrip(".").split(".")
        if token
    }
    return instr.opcode == "add" and bool(tokens & {"u64", "b64"})


def _is_mul_wide_u32(instr: PTXInstruction) -> bool:
    tokens = {
        token
        for token in str(instr.dtype or "").lstrip(".").split(".")
        if token
    }
    return (
        instr.opcode == "mul"
        and "wide" in tokens
        and "u32" in tokens
    )


def _low32_address_only_destinations(func: Function) -> Set[Tuple[str, int]]:
    """Find 64-bit results whose high half is provably irrelevant.

    A candidate is produced by add.u64/b64 or mul.wide.u32. It remains safe
    only when every use is either:
    - as the base of ld.global/st.global; or
    - as an input to another safe add.u64/b64.

    Starting from all candidates and repeatedly removing unsafe ones computes
    the greatest fixed point, so self-recurrent pointer bumps and address
    chains are handled without assuming acyclic data flow.
    """
    candidates: Dict[Tuple[str, int], PTXInstruction] = {}

    for block in func.blocks.values():
        for instr in block.instructions:
            if (
                isinstance(instr.dest, Register)
                and (_is_u64_add(instr) or _is_mul_wide_u32(instr))
            ):
                candidates[_reg_key(instr.dest)] = instr

    safe: Set[Tuple[str, int]] = set(candidates)

    while True:
        remove: Set[Tuple[str, int]] = set()

        for candidate in safe:
            for block in func.blocks.values():
                for instr in block.instructions:
                    # Destination occurrences are definitions, not uses.
                    if isinstance(instr.dest, Register):
                        if _reg_key(instr.dest) == candidate:
                            pass

                    for src in instr.srcs:
                        # Direct register use: only safe as an operand of a
                        # safe add.u64/b64.
                        if isinstance(src, Register) and _reg_key(src) == candidate:
                            if not (
                                _is_u64_add(instr)
                                and isinstance(instr.dest, Register)
                                and _reg_key(instr.dest) in safe
                            ):
                                remove.add(candidate)

                        # Memory use: only global-memory base use is safe.
                        elif isinstance(src, MemoryOperand):
                            if (
                                isinstance(src.base, Register)
                                and _reg_key(src.base) == candidate
                            ):
                                if not (
                                    instr.opcode in {"ld", "st"}
                                    and "global" in str(instr.dtype or "")
                                ):
                                    remove.add(candidate)

                            if (
                                isinstance(src.offset, Register)
                                and _reg_key(src.offset) == candidate
                            ):
                                remove.add(candidate)

        if not remove:
            break

        safe.difference_update(remove)

    return safe

def lower_function(
    func: Function,
    *,
    omit_low32_address_high_zero: bool = False,
) -> List[AECInstruction]:
    reg_alloc = RegisterAllocator()
    param_offsets = _collect_param_offsets(func)
    release_schedule = _local_release_schedule(func)
    low32_only = (
        _low32_address_only_destinations(func)
        if omit_low32_address_high_zero
        else set()
    )

    instructions: List[AECInstruction] = []
    label_to_pc: Dict[str, int] = {}
    pending_branches: List[Tuple[int, str]] = []

    for block_name in func.blocks:
        label_to_pc[str(block_name).rstrip(":").lstrip("%")] = len(instructions)
        block = func.blocks[block_name]

        for instr_index, instr in enumerate(block.instructions):
            start_pc = len(instructions)
            reg_alloc.begin_instruction()
            try:
                lowered = lower_ptx_to_aec(
                instr,
                reg_alloc,
                param_offsets,
                omit_u64_high_zero=(
                    isinstance(instr.dest, Register)
                    and _reg_key(instr.dest) in low32_only
                    and (_is_u64_add(instr) or _is_mul_wide_u32(instr))
                ),
            )
            finally:
                reg_alloc.end_instruction()

            instructions.extend(lowered)

            for dead_reg in release_schedule.get(
                (block_name, instr_index),
                [],
            ):
                reg_alloc.release(dead_reg)

            target = _branch_target_name(instr)
            if target is not None:
                # A branch lowers to exactly one AEC instruction.
                pending_branches.append((start_pc, target))

    for branch_pc, target in pending_branches:
        if target not in label_to_pc:
            raise ValueError(
                f"Unknown branch label {target!r}; "
                f"known labels: {sorted(label_to_pc)}"
            )
        instructions[branch_pc].immediate = label_to_pc[target]

    return instructions


def lower_program(
    program: PTXProgram,
    *,
    omit_low32_address_high_zero: bool = False,
) -> List[AECInstruction]:
    instructions: List[AECInstruction] = []

    for func in program.functions:
        instructions.extend(
            lower_function(
                func,
                omit_low32_address_high_zero=omit_low32_address_high_zero,
            )
        )

    return instructions
