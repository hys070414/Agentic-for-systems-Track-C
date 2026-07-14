import struct
from typing import List

from ir import AECInstruction


# C1 allowed opcodes that carry an immediate in Src2/Imm32.
IMM32_OPCODES = {
    0x0040,  # BR
    0x0041,  # BRX
    0x0055,  # LOADI
    0x0056,  # LOADI64
}


def encode_instruction(instr: AECInstruction) -> bytes:
    """Encode one AEC instruction as four little-endian 32-bit words.

    Layout:
      bits [127:112] opcode
      bits [111:96]  pred/control
      bits [95:80]   dest
      bits [79:64]   src1
      bits [63:32]   src2 or LOADI64 high32
      bits [31:0]    imm32, LOADI64 low32, or src3

    Convention used by this compiler:
      instr.control bit 0 stores pred_neg.
    """
    if not 0 <= instr.opcode <= 0xFFFF:
        raise ValueError(f"opcode out of range: {instr.opcode}")
    if not 0 <= instr.dtype <= 0xF:
        raise ValueError(f"type code out of range: {instr.dtype}")
    if not 0 <= instr.dst <= 0xFFFF:
        raise ValueError(f"destination register out of range: {instr.dst}")
    if not 0 <= instr.src1 <= 0xFFFF:
        raise ValueError(f"source register out of range: {instr.src1}")

    pred_ctrl = (instr.dtype & 0xF) << 3

    # subop for CMPP
    if instr.opcode == 0x0021:
        pred_ctrl |= (instr.modifier & 0x7) << 8

    # memory space for LD/ST
    if instr.opcode in (0x0030, 0x0031):
        pred_ctrl |= (instr.modifier & 0x7) << 11

    # Predication applies to BRX and any predicated instruction.
    if instr.pred != 15:
        if not 0 <= instr.pred <= 7:
            raise ValueError(f"predicate register out of range: {instr.pred}")
        pred_ctrl |= instr.pred & 0x7
        pred_ctrl |= 1 << 15
        if instr.control & 0x1:
            pred_ctrl |= 1 << 14

    if instr.opcode in {0x0040, 0x0041, 0x0055}:
        # BR/BRX/LOADI place their imm32 in bits [31:0] (word0).
        # bits [63:32] (word1/src2) are unused and must be zero.
        word0 = int(instr.immediate) & 0xFFFFFFFF
        word1 = 0
    elif instr.opcode == 0x0056:
        # LOADI64 uses word0 as low32 and word1 as high32.
        imm64 = int(instr.immediate) & 0xFFFFFFFFFFFFFFFF
        word0 = imm64 & 0xFFFFFFFF
        word1 = (imm64 >> 32) & 0xFFFFFFFF
    else:
        word1 = int(instr.src2) & 0xFFFFFFFF
        word0 = int(instr.src3) & 0xFFFFFFFF

    word2 = ((instr.dst & 0xFFFF) << 16) | (instr.src1 & 0xFFFF)
    word3 = ((instr.opcode & 0xFFFF) << 16) | (pred_ctrl & 0xFFFF)

    return struct.pack("<IIII", word0, word1, word2, word3)


def encode_instructions(instructions: List[AECInstruction]) -> bytes:
    if not instructions:
        raise ValueError("AEC program must contain at least one instruction")

    return b"".join(encode_instruction(instr) for instr in instructions)


def encode_program(instructions: List[AECInstruction]) -> bytes:
    """Return a headerless raw AEC 128-bit instruction stream."""
    return encode_instructions(instructions)
