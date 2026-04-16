from transform.transform import SaSSTransform

class OperAggregate(SaSSTransform):
    # Apply operator aggregation on module 
    def apply(self, module):
        TwinIdxes = []
        
        for func in module.functions:
            # Identify the twin-index calculation
            self.IdentifyTwinIdx(func, TwinIdxes)

            # Identify the twin-binary operation
            self.IdentifyTwinBin(func, TwinIdxes)
            
    # Identify the twin-index calculation amd merge the twin-idx related instructions
    def IdentifyTwinIdx(self, func, TwinIdxes):
        for bb in func.blocks:
            Insts = bb.instructions
            # The translated instructions
            TransInsts = []
            # The skipped instructions
            SkipInsts = []

            # Scan the instructions to identify the twin-index pattern
            for i in range(len(Insts)):
                Inst = Insts[i]
                if Inst in SkipInsts:
                    continue
                
                if i < len(Insts) - 1:
                    TwinIdx = self.IsTwinIdxPattern(Inst, Insts[i + 1])
                    if TwinIdx != None:
                        TwinIdxes.append(TwinIdx)
                        TransInsts.append(self.MutateTwinIdxInst(TwinIdx, Inst))
                        # Set skip flag
                        SkipInsts.append(Insts[i + 1])
                        
                        continue

                if Inst.IsNOP():
                    SkipInsts.append(Inst)
                else:
                    # Record the current instruction to translated instructions
                    TransInsts.append(Inst)

            for Inst in SkipInsts:
                # Set opeands as skipped
                Inst.SetSkip()
                
            # Reset the transformed instructions
            bb.instructions = TransInsts
            
    def IsTwinIdxPattern(self, Inst, nextInst):
        if Inst.opcodes[0] == "SHL":
            if nextInst.opcodes[0] == "SHR":
                return Inst.operands[0].Reg + "-" + nextInst.operands[0].Reg

        return None

    # Create the twin-index related pseudo-instruction
    def MutateTwinIdxInst(self, TwinIdx, Inst):
        Inst.TwinIdx = TwinIdx
        return Inst
    
    # Identify the twin-binary operation
    def IdentifyTwinBin(self, func, TwinIdxes):
        for bb in func.blocks:
            Insts = bb.instructions
            # The transslated instructions 
            TransInsts = []
            # The skipped instructions
            SkipInsts = []

            # Scan the instructions to identify the twin-index pattern
            for i in range(len(Insts)):
                Inst = Insts[i]
                if Inst in SkipInsts:
                    continue
                
                if i < len(Insts) - 1 and Inst.IsAddrCompute():
                    NextInst = Insts[i + 1]
                    TwinIdx = self.IsTwinBinPattern(Inst, NextInst)
                    if TwinIdx == None and i < len(Insts) - 2:
                        NextInst = Insts[i + 2]
                        TwinIdx = self.IsTwinBinPattern(Inst, NextInst)

                    if TwinIdx != None and TwinIdx in TwinIdxes:
                        TransInsts.append(self.MutateTwinBinInst(TwinIdx, Inst))
                        # Set skip flag
                        SkipInsts.append(NextInst)
                        
                        continue
                        
                # Record the current instruction to translated instructions
                TransInsts.append(Inst)

            # Set operands as skipped
            for Inst in SkipInsts:
                Inst.SetSkip()
                
            # Reset the transformed instructions
            bb.instructions = TransInsts

    def IsTwinBinPattern(self, Inst, nextInst):
        # Check the match of operator
        if Inst.opcodes[0] != nextInst.opcodes[0]:
            return None

        if len(Inst.opcodes) != 1:
            return None

        # The twin idx instructions should be like:
        # IADD ...
        # IADD.X ...
        if len(nextInst.opcodes) != 2 or nextInst.opcodes[1] != "X":
            return None

        # Check the function argument operands
        if len(Inst.operands) != 3 or len(nextInst.operands) != 3:
            return None

        operand1 = Inst.operands[2]
        operand2 = nextInst.operands[2]
        # The last operand must be function argument and their offsets value have to be continuous
        if not operand1.IsArg or not operand2.IsArg:
            return None
        if operand1.ArgOffset + 4 != operand2.ArgOffset:
            return None

        operand1 = Inst.operands[1]
        operand2= nextInst.operands[1]
        # The 2nd operand must be register operand
        if operand1.IsReg and operand2.IsReg:
            TwinIdx = operand1.Reg + "-" + operand2.Reg
            return TwinIdx

        return None

    def MutateTwinBinInst(self, TwinIdx, Inst):
        Inst.TwinIdx = TwinIdx
        return Inst
