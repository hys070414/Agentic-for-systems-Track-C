from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ir import AECInstruction


OP_ADD = 0x0001
OP_SUB = 0x0002
OP_MUL = 0x0003
OP_MAD = 0x0004
OP_FMA = 0x0005
OP_AND = 0x0010
OP_OR = 0x0011
OP_XOR = 0x0012
OP_SHL = 0x0014
OP_SHR = 0x0015
OP_CMPP = 0x0021
OP_LD = 0x0030
OP_ST = 0x0031
OP_BR = 0x0040
OP_BRX = 0x0041
OP_HALT = 0x0045
OP_CPY = 0x0054
OP_LOADI = 0x0055
OP_LOADI64 = 0x0056

SPACE_GMEM = 0


def _resolve(reg: int, replacements: Dict[int, int]) -> int:
    """Resolve a replacement chain such as R7 -> R5 -> R3."""
    seen: Set[int] = set()
    current = reg

    while current in replacements and current not in seen:
        seen.add(current)
        current = replacements[current]

    return current


def _rewrite_sources(
    instr: AECInstruction,
    replacements: Dict[int, int],
) -> None:
    """Rewrite only fields that are GPR source operands for this opcode."""
    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_CMPP,
    }:
        instr.src1 = _resolve(instr.src1, replacements)
        instr.src2 = _resolve(instr.src2, replacements)

    elif instr.opcode in {OP_MAD, OP_FMA}:
        instr.src1 = _resolve(instr.src1, replacements)
        instr.src2 = _resolve(instr.src2, replacements)
        instr.src3 = _resolve(instr.src3, replacements)

    elif instr.opcode == OP_LD:
        instr.src1 = _resolve(instr.src1, replacements)

    elif instr.opcode == OP_ST:
        instr.src1 = _resolve(instr.src1, replacements)
        instr.src2 = _resolve(instr.src2, replacements)

    elif instr.opcode == OP_CPY:
        # CPY may also read a special-register selector >= 0x0100.
        if 0 < instr.src1 < 0x0100:
            instr.src1 = _resolve(instr.src1, replacements)


def _defines_gpr(instr: AECInstruction) -> bool:
    """Return whether dst is a normal GPR definition."""
    return instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL, OP_MAD, OP_FMA,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_LD, OP_CPY, OP_LOADI, OP_LOADI64,
    }


def _invalidate_register(
    reg: int,
    replacements: Dict[int, int],
    available_loads: Dict[Tuple[int, int, int], int],
) -> None:
    """Invalidate aliases and cached loads affected by a GPR redefinition."""
    replacements.pop(reg, None)

    stale_replacements = [
        old for old, new in replacements.items()
        if old == reg or new == reg
    ]
    for old in stale_replacements:
        replacements.pop(old, None)

    stale_loads = [
        key for key, value_reg in available_loads.items()
        if key[2] == reg or value_reg == reg
    ]
    for key in stale_loads:
        available_loads.pop(key, None)


def eliminate_redundant_global_loads(
    instructions: List[AECInstruction],
    *,
    return_stats: bool = False,
):
    """Remove repeated identical LD.gmem instructions in straight-line code.

    Branch targets are instruction indices. Because removing a load shifts all
    later PCs, this pass also remaps BR/BRX targets to the corresponding new
    instruction index.
    """
    result: List[AECInstruction] = []
    kept_old_indices: List[int] = []
    removed_loads = 0

    # Key: (memory space, dtype, address register)
    available_loads: Dict[Tuple[int, int, int], int] = {}
    replacements: Dict[int, int] = {}

    for old_pc, instr in enumerate(instructions):
        _rewrite_sources(instr, replacements)

        if _defines_gpr(instr):
            _invalidate_register(instr.dst, replacements, available_loads)

        if instr.opcode == OP_LD and instr.modifier == SPACE_GMEM:
            key = (instr.modifier, instr.dtype, instr.src1)
            previous_reg = available_loads.get(key)

            if previous_reg is not None:
                # The new destination has the same value as the earlier load.
                replacements[instr.dst] = _resolve(
                    previous_reg,
                    replacements,
                )
                removed_loads += 1
                continue

            available_loads[key] = instr.dst
            result.append(instr)
            kept_old_indices.append(old_pc)
            continue

        result.append(instr)
        kept_old_indices.append(old_pc)

        # Any store may alias any previous global load, so clear everything.
        if instr.opcode == OP_ST:
            available_loads.clear()

        # Never propagate load reuse across control-flow boundaries.
        if instr.opcode in {OP_BR, OP_BRX, OP_HALT}:
            available_loads.clear()
            replacements.clear()

    # Map every old PC to the first surviving instruction at or after it.
    # This handles both ordinary shifted targets and the conservative case
    # where a branch target itself was an instruction removed by this pass.
    old_count = len(instructions)
    old_to_new = {old_pc: new_pc for new_pc, old_pc in enumerate(kept_old_indices)}

    next_survivor_new_pc = len(result)
    remap = [len(result)] * (old_count + 1)
    for old_pc in range(old_count - 1, -1, -1):
        if old_pc in old_to_new:
            next_survivor_new_pc = old_to_new[old_pc]
        remap[old_pc] = next_survivor_new_pc

    for instr in result:
        if instr.opcode in {OP_BR, OP_BRX}:
            old_target = int(instr.immediate)
            if not 0 <= old_target < old_count:
                raise ValueError(
                    f"Invalid pre-optimization branch target {old_target} "
                    f"for {old_count} instructions"
                )

            new_target = remap[old_target]
            if not 0 <= new_target < len(result):
                raise ValueError(
                    f"Branch target {old_target} has no surviving instruction"
                )
            instr.immediate = new_target

    if return_stats:
        return result, {"redundant_loads_removed": removed_loads}
    return result
