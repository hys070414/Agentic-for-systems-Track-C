from __future__ import annotations

from typing import List, Set

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

VALID_OPCODES = {
    OP_ADD, OP_SUB, OP_MUL, OP_MAD, OP_FMA,
    OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
    OP_CMPP, OP_LD, OP_ST, OP_BR, OP_BRX, OP_HALT,
    OP_CPY, OP_LOADI, OP_LOADI64,
}

VALID_TYPES = {0x0, 0x1, 0x2, 0x3, 0x8, 0xF}
VALID_SPACES = set(range(0, 5))

SPECIAL_REGISTER_SELECTORS = {
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104,
    0x0110, 0x0111, 0x0112, 0x0113,
    0x0120, 0x0121, 0x0122, 0x0123,
}


def _check_gpr(value: int, field: str, pc: int, *, allow_zero: bool = True) -> None:
    minimum = 0 if allow_zero else 1
    if not minimum <= value <= 255:
        raise ValueError(
            f"PC {pc}: {field} must be an AEC GPR in R{minimum}..R255, "
            f"got {value}"
        )


def _pred_enabled(instr: AECInstruction) -> bool:
    return bool(instr.control & 0x8000)


def _validate_sources(instr: AECInstruction, pc: int) -> None:
    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_CMPP,
    }:
        _check_gpr(instr.src1, "src1", pc)
        _check_gpr(instr.src2, "src2", pc)

    elif instr.opcode in {OP_MAD, OP_FMA}:
        _check_gpr(instr.src1, "src1", pc)
        _check_gpr(instr.src2, "src2", pc)
        _check_gpr(instr.src3, "src3", pc)

    elif instr.opcode == OP_LD:
        _check_gpr(instr.src1, "address register", pc, allow_zero=False)

    elif instr.opcode == OP_ST:
        _check_gpr(instr.src1, "address register", pc, allow_zero=False)
        _check_gpr(instr.src2, "store value register", pc)

    elif instr.opcode == OP_CPY:
        if instr.src1 not in SPECIAL_REGISTER_SELECTORS:
            _check_gpr(instr.src1, "src1", pc)


def _validate_opcode_type(instr: AECInstruction, pc: int) -> None:
    """Validate opcode-specific type constraints required by the CModel."""
    if instr.opcode in {OP_LOADI, OP_LOADI64, OP_BR, OP_BRX, OP_HALT}:
        if instr.dtype != 0xF:
            raise ValueError(
                f"PC {pc}: opcode 0x{instr.opcode:04x} requires type none "
                f"(0xF), got 0x{instr.dtype:x}"
            )


def _validate_destination(instr: AECInstruction, pc: int) -> None:
    if instr.opcode == OP_CMPP:
        if not 0 <= instr.dst <= 7:
            raise ValueError(
                f"PC {pc}: CMPP destination must be P0..P7, got {instr.dst}"
            )
        return

    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL, OP_MAD, OP_FMA,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_LD, OP_CPY, OP_LOADI,
    }:
        _check_gpr(instr.dst, "destination", pc, allow_zero=False)

    elif instr.opcode == OP_LOADI64:
        if not 1 <= instr.dst <= 254:
            raise ValueError(
                f"PC {pc}: LOADI64 low register must be R1..R254, "
                f"got R{instr.dst}"
            )


def _validate_predicate(instr: AECInstruction, pc: int) -> None:
    if instr.opcode == OP_BRX or _pred_enabled(instr):
        if not 0 <= instr.pred <= 7:
            raise ValueError(
                f"PC {pc}: predicate index must be P0..P7, got {instr.pred}"
            )


def _validate_branch(instr: AECInstruction, pc: int, count: int) -> None:
    if instr.opcode in {OP_BR, OP_BRX}:
        target = int(instr.immediate)
        if not 0 <= target < count:
            raise ValueError(
                f"PC {pc}: branch target {target} is outside 0..{count - 1}"
            )


def _validate_pair_lowering(
    instructions: List[AECInstruction],
) -> None:
    """Validate adjacent ADD/MUL-low + LOADI-high pair construction."""
    for pc in range(len(instructions) - 1):
        low = instructions[pc]
        high = instructions[pc + 1]

        if (
            low.opcode in {OP_ADD, OP_MUL}
            and high.opcode == OP_LOADI
            and high.immediate == 0
            and high.dst == low.dst + 1
        ):
            if low.dst >= 255:
                raise ValueError(
                    f"PC {pc}: 64-bit pair low register cannot be R{low.dst}"
                )
            if low.dst % 2 != 0:
                raise ValueError(
                    f"PC {pc}: 64-bit pair low register should be even, "
                    f"got R{low.dst}"
                )



# Physical allocation policy:
#   R1..R239   are preserved for ordinary lowered values.
#   R240..R247 are per-instruction spill value scratch registers.
#   R255       is the local-memory byte-address scratch register.
#
# Virtual registers >= R240 are assigned 4-byte slots in per-thread lmem.
PHYSICAL_GPR_LIMIT = 239
SPILL_VALUE_SCRATCH = tuple(range(240, 248))
SPILL_ADDR_SCRATCH = 255
LMEM_SPACE = 3
TYPE_B32 = 0x0
TYPE_NONE = 0xF
MAX_LMEM_BYTES = 4096

_last_spill_loads = 0
_last_spill_stores = 0


def _clone_instruction(instr: AECInstruction) -> AECInstruction:
    return AECInstruction(
        opcode=instr.opcode,
        dtype=instr.dtype,
        pred=instr.pred,
        control=instr.control,
        dst=instr.dst,
        src1=instr.src1,
        src2=instr.src2,
        src3=instr.src3,
        immediate=instr.immediate,
        modifier=instr.modifier,
        original=instr.original,
    )


def _is_virtual_gpr(value: int) -> bool:
    return isinstance(value, int) and value >= 1


def _is_spilled_gpr(value: int) -> bool:
    return _is_virtual_gpr(value) and value > PHYSICAL_GPR_LIMIT


def _source_fields(instr: AECInstruction) -> List[str]:
    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_CMPP,
    }:
        return ["src1", "src2"]

    if instr.opcode in {OP_MAD, OP_FMA}:
        return ["src1", "src2", "src3"]

    if instr.opcode == OP_LD:
        return ["src1"]

    if instr.opcode == OP_ST:
        return ["src1", "src2"]

    if instr.opcode == OP_CPY:
        if instr.src1 in SPECIAL_REGISTER_SELECTORS:
            return []
        return ["src1"]

    return []


def _destination_registers(instr: AECInstruction) -> List[int]:
    if instr.opcode == OP_CMPP:
        return []

    if instr.opcode in {
        OP_ADD, OP_SUB, OP_MUL, OP_MAD, OP_FMA,
        OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
        OP_LD, OP_CPY, OP_LOADI,
    }:
        return [instr.dst]

    if instr.opcode == OP_LOADI64:
        return [instr.dst, instr.dst + 1]

    return []


def _slot_offset(vreg: int) -> int:
    offset = (vreg - (PHYSICAL_GPR_LIMIT + 1)) * 4
    if offset < 0 or offset + 4 > MAX_LMEM_BYTES:
        raise ValueError(
            f"Spill slot for virtual R{vreg} exceeds {MAX_LMEM_BYTES} bytes "
            f"of local memory"
        )
    return offset


def _make_loadi(dst: int, value: int) -> AECInstruction:
    return AECInstruction(
        opcode=OP_LOADI,
        dtype=TYPE_NONE,
        pred=15,
        control=0,
        dst=dst,
        src1=0,
        src2=0,
        src3=0,
        immediate=value & 0xFFFFFFFF,
        modifier=0,
    )


def _make_spill_load(
    scratch: int,
    vreg: int,
    template: AECInstruction,
) -> List[AECInstruction]:
    return [
        _make_loadi(SPILL_ADDR_SCRATCH, _slot_offset(vreg)),
        AECInstruction(
            opcode=OP_LD,
            dtype=TYPE_B32,
            pred=template.pred,
            control=template.control,
            dst=scratch,
            src1=SPILL_ADDR_SCRATCH,
            src2=0,
            src3=0,
            immediate=0,
            modifier=LMEM_SPACE,
            original=template.original,
        ),
    ]


def _make_spill_store(
    scratch: int,
    vreg: int,
    template: AECInstruction,
) -> List[AECInstruction]:
    return [
        _make_loadi(SPILL_ADDR_SCRATCH, _slot_offset(vreg)),
        AECInstruction(
            opcode=OP_ST,
            dtype=TYPE_B32,
            pred=template.pred,
            control=template.control,
            dst=0,
            src1=SPILL_ADDR_SCRATCH,
            src2=scratch,
            src3=0,
            immediate=0,
            modifier=LMEM_SPACE,
            original=template.original,
        ),
    ]


def _rewrite_with_spills(
    instructions: List[AECInstruction],
) -> tuple[List[AECInstruction], int, int]:
    """Rewrite virtual GPRs above R239 through per-thread local memory."""
    rewritten: List[AECInstruction] = []
    old_pc_to_new_pc: dict[int, int] = {}
    branch_fixups: List[tuple[AECInstruction, int]] = []

    spill_loads = 0
    spill_stores = 0

    for old_pc, original in enumerate(instructions):
        old_pc_to_new_pc[old_pc] = len(rewritten)
        core = _clone_instruction(original)

        source_fields = _source_fields(core)
        destinations = _destination_registers(core)

        spilled_values: List[int] = []
        for field in source_fields:
            value = getattr(core, field)
            if _is_spilled_gpr(value) and value not in spilled_values:
                spilled_values.append(value)

        for value in destinations:
            if _is_spilled_gpr(value) and value not in spilled_values:
                spilled_values.append(value)

        if len(spilled_values) > len(SPILL_VALUE_SCRATCH):
            raise ValueError(
                f"PC {old_pc}: instruction needs {len(spilled_values)} spill "
                f"scratch registers, only {len(SPILL_VALUE_SCRATCH)} available"
            )

        scratch_for = {
            vreg: SPILL_VALUE_SCRATCH[index]
            for index, vreg in enumerate(spilled_values)
        }

        # Load each spilled source once. Read-modify-write destinations reuse
        # the same scratch value.
        loaded: Set[int] = set()
        for field in source_fields:
            value = getattr(core, field)
            if not _is_spilled_gpr(value):
                continue
            if value not in loaded:
                rewritten.extend(_make_spill_load(scratch_for[value], value, original))
                spill_loads += 1
                loaded.add(value)
            setattr(core, field, scratch_for[value])

        # Rewrite destinations. LOADI64 is split into two scalar LOADI
        # instructions when either half is spilled, because its implicit
        # consecutive-register destination cannot target unrelated spill slots.
        if core.opcode == OP_LOADI64 and any(
            _is_spilled_gpr(value) for value in destinations
        ):
            low_vreg, high_vreg = destinations
            low_scratch = scratch_for.get(low_vreg, low_vreg)
            high_scratch = scratch_for.get(high_vreg, high_vreg)
            low_value = int(core.immediate) & 0xFFFFFFFF
            high_value = (int(core.immediate) >> 32) & 0xFFFFFFFF

            low_loadi = AECInstruction(
                opcode=OP_LOADI,
                dtype=TYPE_NONE,
                pred=core.pred,
                control=core.control,
                dst=low_scratch,
                src1=0,
                src2=0,
                src3=0,
                immediate=low_value,
                modifier=0,
                original=core.original,
            )
            high_loadi = AECInstruction(
                opcode=OP_LOADI,
                dtype=TYPE_NONE,
                pred=core.pred,
                control=core.control,
                dst=high_scratch,
                src1=0,
                src2=0,
                src3=0,
                immediate=high_value,
                modifier=0,
                original=core.original,
            )

            rewritten.append(low_loadi)
            if _is_spilled_gpr(low_vreg):
                rewritten.extend(
                    _make_spill_store(low_scratch, low_vreg, original)
                )
                spill_stores += 1

            rewritten.append(high_loadi)
            if _is_spilled_gpr(high_vreg):
                rewritten.extend(
                    _make_spill_store(high_scratch, high_vreg, original)
                )
                spill_stores += 1
            continue

        spilled_destinations: List[tuple[int, int]] = []
        if destinations:
            # All ordinary instructions have one explicit GPR destination.
            value = destinations[0]
            if _is_spilled_gpr(value):
                core.dst = scratch_for[value]
                spilled_destinations.append((value, scratch_for[value]))

        if core.opcode in {OP_BR, OP_BRX}:
            branch_fixups.append((core, int(core.immediate)))

        rewritten.append(core)

        for vreg, scratch in spilled_destinations:
            rewritten.extend(_make_spill_store(scratch, vreg, original))
            spill_stores += 1

    # A branch must land at the first inserted preload for its old target.
    for branch, old_target in branch_fixups:
        if old_target not in old_pc_to_new_pc:
            raise ValueError(
                f"Branch target {old_target} is outside original instruction range"
            )
        branch.immediate = old_pc_to_new_pc[old_target]

    return rewritten, spill_loads, spill_stores


def _validate_allocated(instructions: List[AECInstruction]) -> None:
    if not instructions:
        raise ValueError("Cannot allocate registers for an empty program")

    count = len(instructions)

    for pc, instr in enumerate(instructions):
        if instr.opcode not in VALID_OPCODES:
            raise ValueError(
                f"PC {pc}: unsupported C1 opcode 0x{instr.opcode:04x}"
            )

        if instr.dtype not in VALID_TYPES:
            raise ValueError(
                f"PC {pc}: invalid C1 type code 0x{instr.dtype:x}"
            )

        if instr.opcode in {OP_LD, OP_ST}:
            if instr.modifier not in VALID_SPACES:
                raise ValueError(
                    f"PC {pc}: invalid C1 memory-space code {instr.modifier}"
                )

        _validate_opcode_type(instr, pc)
        _validate_destination(instr, pc)
        _validate_sources(instr, pc)
        _validate_predicate(instr, pc)
        _validate_branch(instr, pc, count)

    _validate_pair_lowering(instructions)


def allocate_registers(instructions: List[AECInstruction]) -> None:
    """Apply local-memory spill rewriting, then validate physical AEC GPRs."""
    global _last_spill_loads, _last_spill_stores

    rewritten, loads, stores = _rewrite_with_spills(instructions)
    instructions[:] = rewritten

    _last_spill_loads = loads
    _last_spill_stores = stores

    _validate_allocated(instructions)


def physical_registers_used(
    instructions: List[AECInstruction],
) -> Set[int]:
    """Return the set of concrete GPR numbers referenced by the program."""
    used: Set[int] = set()

    for instr in instructions:
        if instr.opcode != OP_CMPP and 1 <= instr.dst <= 255:
            if instr.opcode not in {OP_ST, OP_BR, OP_BRX, OP_HALT}:
                used.add(instr.dst)

        if instr.opcode == OP_LOADI64 and instr.dst < 255:
            used.add(instr.dst + 1)

        if instr.opcode in {
            OP_ADD, OP_SUB, OP_MUL,
            OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR,
            OP_CMPP,
        }:
            used.update(r for r in (instr.src1, instr.src2) if 1 <= r <= 255)

        elif instr.opcode in {OP_MAD, OP_FMA}:
            used.update(
                r for r in (instr.src1, instr.src2, instr.src3)
                if 1 <= r <= 255
            )

        elif instr.opcode == OP_LD:
            if 1 <= instr.src1 <= 255:
                used.add(instr.src1)

        elif instr.opcode == OP_ST:
            used.update(r for r in (instr.src1, instr.src2) if 1 <= r <= 255)

        elif instr.opcode == OP_CPY:
            if 1 <= instr.src1 <= 255:
                used.add(instr.src1)

    return used


def allocation_stats(instructions: List[AECInstruction]) -> dict:
    """Return physical-register and actual spill-instruction statistics."""
    used = physical_registers_used(instructions)
    return {
        "physical_registers": len(used),
        "max_physical_register": max(used, default=0),
        "spill_loads": _last_spill_loads,
        "spill_stores": _last_spill_stores,
    }
