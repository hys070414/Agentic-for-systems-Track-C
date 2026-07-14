import struct


OPCODE_NAMES = {
    0x0001: "ADD",
    0x0002: "SUB",
    0x0003: "MUL",
    0x0004: "MAD",
    0x0005: "FMA",
    0x0010: "AND",
    0x0011: "OR",
    0x0012: "XOR",
    0x0014: "SHL",
    0x0015: "SHR",
    0x0021: "CMPP",
    0x0030: "LD",
    0x0031: "ST",
    0x0040: "BR",
    0x0041: "BRX",
    0x0045: "HALT",
    0x0054: "CPY",
    0x0055: "LOADI",
    0x0056: "LOADI64",
}

TYPE_NAMES = {
    0x0: "b32",
    0x1: "b64",
    0x2: "u32",
    0x3: "s32",
    0x8: "f32",
    0xF: "none",
}

MEMORY_SPACE_NAMES = {
    0: "gmem",
    1: "smem",
    2: "cmem",
    3: "lmem",
    4: "pmem",
}

CMP_OP_NAMES = {
    0: "eq",
    1: "ne",
    2: "lt",
    3: "le",
    4: "gt",
    5: "ge",
}

SPECIAL_REGISTER_NAMES = {
    0x0100: "%tid.x",
    0x0101: "%ntid.x",
    0x0102: "%ctaid.x",
    0x0103: "%nctaid.x",
    0x0104: "%laneid",
    0x0110: "%tid.y",
    0x0111: "%ntid.y",
    0x0112: "%ctaid.y",
    0x0113: "%nctaid.y",
    0x0120: "%tid.z",
    0x0121: "%ntid.z",
    0x0122: "%ctaid.z",
    0x0123: "%nctaid.z",
}


def _reg(value: int) -> str:
    return f"R{value}"


def decode_instruction(data: bytes, offset: int) -> str:
    word0, word1, word2, word3 = struct.unpack_from("<IIII", data, offset)

    opcode = (word3 >> 16) & 0xFFFF
    pred_ctrl = word3 & 0xFFFF
    dest = (word2 >> 16) & 0xFFFF
    src1 = word2 & 0xFFFF
    src2 = word1
    src3 = word0

    pred = pred_ctrl & 0x7
    dtype = (pred_ctrl >> 3) & 0xF
    subop = (pred_ctrl >> 8) & 0x7
    space = (pred_ctrl >> 11) & 0x7
    pred_neg = (pred_ctrl >> 14) & 0x1
    pred_en = (pred_ctrl >> 15) & 0x1

    op_name = OPCODE_NAMES.get(opcode, f"0x{opcode:04x}")
    type_name = TYPE_NAMES.get(dtype, f"type0x{dtype:x}")
    suffix = "" if dtype == 0xF else f".{type_name}"

    pred_text = ""
    if pred_en:
        pred_text = f"@{'!' if pred_neg else ''}P{pred} "

    if opcode == 0x0045:
        return f"{pred_text}HALT"

    if opcode == 0x0040:
        return f"BR {word0}"

    if opcode == 0x0041:
        branch_pred = f"{'!' if pred_neg else ''}P{pred}"
        return f"BRX {branch_pred}, {word0}"

    if opcode == 0x0055:
        return f"{pred_text}LOADI{suffix} {_reg(dest)}, 0x{word0:08x}"

    if opcode == 0x0056:
        imm64 = (word1 << 32) | word0
        return f"{pred_text}LOADI64{suffix} {_reg(dest)}, 0x{imm64:016x}"

    if opcode == 0x0054:
        source = SPECIAL_REGISTER_NAMES.get(src1, _reg(src1))
        return f"{pred_text}CPY{suffix} {_reg(dest)}, {source}"

    if opcode == 0x0021:
        cmp_name = CMP_OP_NAMES.get(subop, str(subop))
        return (
            f"{pred_text}CMPP{suffix}.{cmp_name} "
            f"P{dest}, {_reg(src1)}, {_reg(src2)}"
        )

    if opcode == 0x0030:
        space_name = MEMORY_SPACE_NAMES.get(space, str(space))
        return (
            f"{pred_text}LD.{space_name}{suffix} "
            f"{_reg(dest)}, [{_reg(src1)}]"
        )

    if opcode == 0x0031:
        space_name = MEMORY_SPACE_NAMES.get(space, str(space))
        return (
            f"{pred_text}ST.{space_name}{suffix} "
            f"[{_reg(src1)}], {_reg(src2)}"
        )

    if opcode in (0x0004, 0x0005):
        return (
            f"{pred_text}{op_name}{suffix} {_reg(dest)}, "
            f"{_reg(src1)}, {_reg(src2)}, {_reg(src3)}"
        )

    if opcode in {
        0x0001, 0x0002, 0x0003,
        0x0010, 0x0011, 0x0012, 0x0014, 0x0015,
    }:
        return (
            f"{pred_text}{op_name}{suffix} {_reg(dest)}, "
            f"{_reg(src1)}, {_reg(src2)}"
        )

    return (
        f"{pred_text}{op_name}{suffix} "
        f"dst={dest}, src1={src1}, src2=0x{src2:08x}, "
        f"immext=0x{src3:08x}"
    )


def disassemble_aecbin(data: bytes) -> str:
    if not data:
        raise ValueError("Empty .aecbin file")

    if len(data) % 16 != 0:
        raise ValueError(
            f"Invalid .aecbin size: {len(data)} bytes; "
            "expected a multiple of 16"
        )

    result = ["=== Code ==="]

    for index in range(len(data) // 16):
        result.append(f"{index:4d}: {decode_instruction(data, index * 16)}")

    return "\n".join(result)
