from sir.module import Module
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.defuse import DefUse
from sir.function import Function
from sir.controlcode import ControlCode
from sir.operand import Operand


class BranchDivergence:
    def apply(self, module):
        for func in module.functions:
            
            #TODO :: Traverse the CFG to obtain a list of basic blocks to work with 
                #Hint: The entry block is the first block in the function
                #Hint: See function.py and basicblock.py
            
            # List to collect all branch instructions that are thread ID dependent
            branch_divergence_inst_ids = []

            #TODO :: Construct global def-use table
                #Hint: See instruction.py and operand.py
                #Hint: Instruction.GetDef() and Instruction.GetUse() return registers
                #Hint: Tricky: a register name can store multiple potential definitions from multiple predecessor basic blocks 
                #Hint: Use an iterative approach to handle loops.
            
            # Keep track of which registers have thread ID dependency at the end of each block
            # Key: BasicBlock, Value: set of register names that are tainted
            taint_at_block_exit = {}
            for bb in func.blocks:
                taint_at_block_exit[bb] = set()

            #TODO :: Find thread ID dependent instructions 
                #Hint: opcode "S2R" is used to load thread ID into a register

            #TODO :: Implement an iterative algorithm to propagate thread ID dependency across instructions

            # Keep iterating until we reach a stable state (no more changes)
            # This handles loops where taint needs to propagate through multiple iterations
            still_changing = True
            while still_changing:
                still_changing = False
                
                # Go through each basic block and analyze it
                for bb in func.blocks:
                    # Start with the taint from predecessor blocks
                    # Merge taint from all incoming edges
                    current_taint = set()
                    for pred_bb in bb._preds:
                        current_taint = current_taint | taint_at_block_exit[pred_bb]

                    # Now process each instruction in order
                    # Track taint as we go through the block sequentially
                    for inst in bb.instructions:
                        # Check if this instruction is tainted (uses thread ID data)
                        inst_is_tainted = False
                        
                        # Case 1: Direct S2R instruction that loads thread ID
                        # S2R is special register read, operand[1] tells us what special reg
                        if len(inst.opcodes) > 0 and inst.opcodes[0] == 'S2R':
                            if len(inst.operands) > 1 and inst.operands[1].IsThreadIdx:
                                inst_is_tainted = True
                        
                        # Case 2: Instruction guarded by a tainted predicate register
                        # If pflag (predicate flag) is in tainted registers, this instruction is data dependent
                        if not inst_is_tainted:
                            if inst.pflag is not None and inst.pflag in current_taint:
                                inst_is_tainted = True
                        
                        # Case 3: Instruction uses a register that's already tainted
                        # Look at all input operands (uses) of the instruction
                        if not inst_is_tainted:
                            for used_op in inst.GetUses():
                                # Only care about register operands, not immediates or other types
                                if used_op.IsReg and used_op.Reg in current_taint:
                                    inst_is_tainted = True
                                    break

                        # If this is a tainted branch instruction, it's a divergence point
                        # Different threads may take different branches based on thread ID
                        if inst_is_tainted and inst.IsBranch():
                            branch_divergence_inst_ids.append(inst.id)

                        # Now handle the output of this instruction
                        # Update taint info for registers this instruction defines
                        if len(inst.operands) == 0:
                            continue

                        # Get what register this instruction writes to
                        def_op = inst.GetDef()
                        if not def_op.IsReg:
                            continue

                        # Update the taint tracking based on whether instruction is tainted
                        output_reg = def_op.Reg
                        if inst_is_tainted:
                            # Tainted instruction taints its output
                            # So dependent instructions will also be tainted
                            current_taint.add(output_reg)
                        else:
                            # Non-tainted instruction writes clean data
                            # This untaints the register even if it was tainted before
                            current_taint.discard(output_reg)

                    # Check if block's exit taint changed
                    # If yes, we need to run another iteration for successor blocks
                    if current_taint != taint_at_block_exit[bb]:
                        taint_at_block_exit[bb] = current_taint
                        still_changing = True

            #TODO :: Update the list below with the branch divergence instruction IDs   
            # Please donot change the name of this function
            # module.branch_divergence_insts.append(...)
            
            # Add all found branch divergence points to the module result
            # Remove duplicates by converting to set then back to list
            unique_divergence_ids = list(set(branch_divergence_inst_ids))
            module.branch_divergence_insts.extend(unique_divergence_ids)