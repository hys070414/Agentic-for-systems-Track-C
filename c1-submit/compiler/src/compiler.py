from ptx_parser import parse_ptx
from instruction_lowering import lower_program
from optimizer import optimize_program
from register_allocator import allocate_registers, allocation_stats
from instruction_scheduler import schedule_instructions
from binary_encoder import encode_program
from memory_optimizer import eliminate_redundant_global_loads


def compile_ptx_to_aecbin(
    source: str,
    opt_level: str = "O2",
    *,
    return_stats: bool = False,
):
    program = parse_ptx(source)

    stats = {
        "cse_removed": 0,
        "dce_removed": 0,
        "licm_moved": 0,
        "strength_reduced": 0,
        "constants_propagated": 0,
        "constants_folded": 0,
        "algebraic_simplified": 0,
        "redundant_loads_removed": 0,
        "spill_loads": 0,
        "spill_stores": 0,
        "scheduler": "none",
        "low32_address_high_writes_omitted": 0,
    }

    if opt_level != "O0":
        stats.update(optimize_program(program, opt_level))

    omit_low32_address_high_zero = opt_level in ("O2", "O3")

    instructions = lower_program(
        program,
        omit_low32_address_high_zero=omit_low32_address_high_zero,
    )

    if opt_level in ("O2", "O3"):
        instructions, mem_stats = eliminate_redundant_global_loads(
            instructions,
            return_stats=True,
        )
        stats.update(mem_stats)
        instructions = schedule_instructions(instructions)
        stats["scheduler"] = "list"

    allocate_registers(instructions)
    stats.update(allocation_stats(instructions))

    aecbin = encode_program(instructions)
    stats["num_aec_instructions"] = len(instructions)

    if return_stats:
        return aecbin, stats
    return aecbin
