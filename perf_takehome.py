"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class VReg:
    """A virtual register representing a value, not a physical location."""
    _counter = 0

    def __init__(self, name_hint="", size=1, pinned_addr=None):
        self.id = VReg._counter
        VReg._counter += 1
        self.name_hint = name_hint
        self.size = size  # 1 for scalar, VLEN for vector
        self.pinned_addr = pinned_addr  # If set, must use this physical address

    def __repr__(self):
        if self.name_hint:
            return f"v{self.id}_{self.name_hint}"
        return f"v{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, VReg):
            return self.id == other.id
        return False


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vregs = {}  # name -> VReg for named vregs

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def new_vreg(self, name_hint="", size=1):
        """Create a new unique virtual register."""
        return VReg(name_hint=name_hint, size=size)

    def new_vreg_vec(self, name_hint=""):
        """Create a new unique vector virtual register."""
        return VReg(name_hint=name_hint, size=VLEN)

    def pinned_vreg(self, name, size=1):
        """Create or get a pinned virtual register (allocated to fixed physical address)."""
        if name not in self.vregs:
            addr = self.alloc_scratch(name, size)
            vreg = VReg(name_hint=name, size=size, pinned_addr=addr)
            self.vregs[name] = vreg
        return self.vregs[name]

    def pinned_const(self, val, name=None):
        """Get a pinned vreg for a constant value."""
        if val not in self.const_map:
            vreg_name = name or f"const_{val}"
            addr = self.alloc_scratch(vreg_name)
            self.add("load", ("const", addr, val))
            vreg = VReg(name_hint=vreg_name, size=1, pinned_addr=addr)
            self.const_map[val] = vreg
        return self.const_map[val]

    def allocate_vregs(self, bundles):
        """
        Allocate physical scratch addresses for virtual registers.
        Uses liveness analysis to reuse scratch space.

        Args:
            bundles: List of bundles, where each bundle is a list of slots.
                     Each bundle represents one cycle.

        Returns:
            List of bundles with physical addresses instead of VRegs.
        """
        # Step 1: Compute liveness - find last cycle where each vreg is used
        last_use = {}  # vreg -> cycle index of last use

        def collect_vregs(args):
            """Collect all VRegs from instruction arguments."""
            for a in args:
                if isinstance(a, VReg) and a.pinned_addr is None:
                    yield a

        for cycle_idx, bundle in enumerate(bundles):
            for (engine, args) in bundle:
                if engine == "debug":
                    continue
                for vreg in collect_vregs(args):
                    last_use[vreg] = cycle_idx

        # Step 2: Allocate with reuse based on cycle-level liveness
        vreg_to_addr = {}
        free_pool = defaultdict(list)  # size -> list of free addresses

        def get_addr(vreg):
            if isinstance(vreg, VReg):
                if vreg.pinned_addr is not None:
                    return vreg.pinned_addr
                if vreg not in vreg_to_addr:
                    if free_pool[vreg.size]:
                        addr = free_pool[vreg.size].pop()
                    else:
                        addr = self.alloc_scratch(f"pool_{vreg.size}_{len(vreg_to_addr)}", vreg.size)
                    vreg_to_addr[vreg] = addr
                return vreg_to_addr[vreg]
            return vreg

        def free_dead_vregs(cycle_idx):
            """Return addresses of vregs whose last use was this cycle."""
            dead = [v for v, last in last_use.items() if last == cycle_idx and v in vreg_to_addr]
            for vreg in dead:
                addr = vreg_to_addr[vreg]
                free_pool[vreg.size].append(addr)

        def rewrite_slot(slot):
            engine, args = slot
            if engine == "debug":
                return slot
            new_args = tuple(get_addr(a) if isinstance(a, (VReg, int)) else a for a in args)
            return (engine, new_args)

        physical_bundles = []
        for cycle_idx, bundle in enumerate(bundles):
            physical_bundle = [rewrite_slot(slot) for slot in bundle]
            physical_bundles.append(physical_bundle)
            free_dead_vregs(cycle_idx)

        return physical_bundles

    def build(self, bundles: list[list[tuple[Engine, tuple]]]):
        """
        Convert bundles of slots into instruction format.

        Args:
            bundles: List of bundles, where each bundle is a list of (engine, args) slots.

        Returns:
            List of instruction dicts: [{engine: [slot, ...], ...}, ...]
        """
        instrs = []
        for bundle in bundles:
            instr = defaultdict(list)
            for engine, args in bundle:
                instr[engine].append(args)
            instrs.append(dict(instr))
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_vhash(self, val_in, hash_const_vecs):
        """Vectorized hash - operates on VLEN elements at once.
        Uses virtual registers. Returns (slots, val_out) where val_out is the result vreg."""
        slots = []
        val_vec = val_in

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec = hash_const_vecs[hi * 2]
            const3_vec = hash_const_vecs[hi * 2 + 1]

            # Create fresh vregs for this stage's outputs
            tmp1 = self.new_vreg_vec(f"hash{hi}_t1")
            tmp2 = self.new_vreg_vec(f"hash{hi}_t2")
            val_out = self.new_vreg_vec(f"hash{hi}_out")

            # tmp1 = op1(val, const1) and tmp2 = op3(val, const3) - independent
            slots.append(("valu", (op1, tmp1, val_vec, const1_vec)))
            slots.append(("valu", (op3, tmp2, val_vec, const3_vec)))
            slots.append(("valu", (op2, val_out, tmp1, tmp2)))

            val_vec = val_out  # Chain to next stage

        return slots, val_vec

    def build_gather(self, addr_vec, name_hint="gather"):
        """Gather VLEN values from non-contiguous addresses into a vector.
        Returns (slots, dest_vreg) where dest_vreg is the result."""
        dest_vec = self.new_vreg_vec(name_hint)
        slots = []
        for i in range(VLEN):
            slots.append(("load", ("load_offset", dest_vec, addr_vec, i)))
        return slots, dest_vec

    def setup_kernel_scratch_and_constants(self):
        """
        Allocate scratch space and set up initial constants.
        Returns tuple of (tmp1, tmp2, tmp3, zero_const, one_const, two_const).
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Scratch space addresses for kernel parameters
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pre-load all hash function constants to avoid loading them in the hot loop
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            self.scratch_const(val1)
            self.scratch_const(val3)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        return tmp1, tmp2, tmp3, zero_const, one_const, two_const

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized kernel using virtual registers.
        Each write creates a fresh vreg (SSA form).
        """
        # Setup phase uses physical addresses for parameters loaded from memory
        tmp1, tmp2, tmp3, zero_const, one_const, two_const = (
            self.setup_kernel_scratch_and_constants()
        )

        body = []  # array of (engine, args) slots with virtual registers

        # Pinned vregs for broadcast constants (allocated once, used throughout)
        one_vec = self.pinned_vreg("one_vec", VLEN)
        two_vec = self.pinned_vreg("two_vec", VLEN)
        n_nodes_vec = self.pinned_vreg("n_nodes_vec", VLEN)
        forest_p_vec = self.pinned_vreg("forest_p_vec", VLEN)

        body.append(("valu", ("vbroadcast", one_vec, one_const)))
        body.append(("valu", ("vbroadcast", two_vec, two_const)))
        body.append(("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"])))
        body.append(("valu", ("vbroadcast", forest_p_vec, self.scratch["forest_values_p"])))

        # Pre-broadcast all 12 hash constants (pinned since used every iteration)
        hash_const_vecs = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            const1_vec = self.pinned_vreg(f"hash_c1_{hi}_vec", VLEN)
            const3_vec = self.pinned_vreg(f"hash_c3_{hi}_vec", VLEN)
            body.append(("valu", ("vbroadcast", const1_vec, self.scratch_const(val1))))
            body.append(("valu", ("vbroadcast", const3_vec, self.scratch_const(val3))))
            hash_const_vecs.append(const1_vec)
            hash_const_vecs.append(const3_vec)

        n_vectors = batch_size // VLEN

        for rnd in range(rounds):
            for vi in range(n_vectors):
                offset = vi * VLEN
                offset_const = self.scratch_const(offset)

                # Fresh vregs for this iteration's computations
                idx_base = self.new_vreg(f"idx_base_r{rnd}_v{vi}")
                val_base = self.new_vreg(f"val_base_r{rnd}_v{vi}")

                # Compute base addresses
                body.append(("alu", ("+", idx_base, self.scratch["inp_indices_p"], offset_const)))
                body.append(("alu", ("+", val_base, self.scratch["inp_values_p"], offset_const)))

                # Load indices and values into fresh vregs
                idx_loaded = self.new_vreg_vec(f"idx_r{rnd}_v{vi}")
                val_loaded = self.new_vreg_vec(f"val_r{rnd}_v{vi}")
                body.append(("load", ("vload", idx_loaded, idx_base)))
                body.append(("load", ("vload", val_loaded, val_base)))

                # Compute gather addresses: addr = forest_p + idx
                addr_vec = self.new_vreg_vec(f"addr_r{rnd}_v{vi}")
                body.append(("valu", ("+", addr_vec, forest_p_vec, idx_loaded)))

                # Gather node values from tree
                gather_slots, node_val = self.build_gather(addr_vec, f"node_r{rnd}_v{vi}")
                body.extend(gather_slots)

                # val = val ^ node_val
                val_xored = self.new_vreg_vec(f"xor_r{rnd}_v{vi}")
                body.append(("valu", ("^", val_xored, val_loaded, node_val)))

                # val = myhash(val)
                hash_slots, val_hashed = self.build_vhash(val_xored, hash_const_vecs)
                body.extend(hash_slots)

                # idx = 2*idx + 1 + (val & 1)
                parity = self.new_vreg_vec(f"parity_r{rnd}_v{vi}")
                idx_doubled_plus1 = self.new_vreg_vec(f"idx2p1_r{rnd}_v{vi}")
                idx_next = self.new_vreg_vec(f"idx_next_r{rnd}_v{vi}")

                body.append(("valu", ("&", parity, val_hashed, one_vec)))
                body.append(("valu", ("multiply_add", idx_doubled_plus1, idx_loaded, two_vec, one_vec)))
                body.append(("valu", ("+", idx_next, idx_doubled_plus1, parity)))

                # idx = idx * (idx < n_nodes) -- wraps to 0 if out of bounds
                in_bounds = self.new_vreg_vec(f"inbounds_r{rnd}_v{vi}")
                idx_wrapped = self.new_vreg_vec(f"idx_wrap_r{rnd}_v{vi}")
                body.append(("valu", ("<", in_bounds, idx_next, n_nodes_vec)))
                body.append(("valu", ("*", idx_wrapped, idx_next, in_bounds)))

                # Store results
                body.append(("store", ("vstore", idx_base, idx_wrapped)))
                body.append(("store", ("vstore", val_base, val_hashed)))

        # Convert flat slot list to bundles (one slot per bundle for now)
        # A scheduler would pack multiple slots into fewer bundles
        bundles = [[slot] for slot in body]

        # Allocate physical addresses for virtual registers
        physical_bundles = self.allocate_vregs(bundles)

        body_instrs = self.build(physical_bundles)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
