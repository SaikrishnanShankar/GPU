from transform.transform import SaSSTransform
from collections import deque

class TypeAnalysis(SaSSTransform):
    def apply(self, module):
        for func in module.functions:
            self.ProcessFunc(func)


    def ProcessFunc(self, function):
        WorkList = self.TraverseCFG(function)

        NewRegsMap = {}
        RegsMap = {}

        NewRegsRevMap = {}
        RegsRevMap = {}

        for BB in WorkList:
            NewRegsMap[BB] = {}
            RegsMap[BB] = {}
            NewRegsRevMap[BB] = {}
            RegsRevMap[BB] = {}

        Changed = True
        
        while Changed:
            Changed = False

            for BB in WorkList:
                Changed |= self.ProcessBB(BB, NewRegsMap, RegsMap)
            
            for BB in reversed(WorkList):
                Changed |= self.ProcessBBRev(BB, NewRegsRevMap, RegsRevMap)

            # for BB in WorkList:
            #     for Inst in BB.instructions:
            #         print(Inst._InstContent+" => ", end="")
            #         for Operand in Inst.operands:
            #             print(Operand._TypeDesc+", ",end="")
            #         print("")

            # print(".")

        # print("done")


    def TraverseCFG(self, function):
        EntryBB = function.blocks[0]
        Visited = set()
        Queue = deque([EntryBB])
        WorkList = []

        while Queue:
            CurrBB = Queue.popleft()
            
            if CurrBB in Visited:
                continue

            Visited.add(CurrBB)
            WorkList.append(CurrBB)

            for SuccBB in CurrBB._succs:
                if SuccBB not in Visited:
                    Queue.append(SuccBB)

        return WorkList
        

    def ProcessBB(self, BB, NewRegsMap, RegsMap):
            
        NewRegs = {}

        for Inst in BB.instructions:

            if Inst.IsStore():
                self.HandleStore(Inst, RegsMap[BB], NewRegs)

            Inst.ResolveType()

            self.UpdateNewRegsMap(NewRegs, Inst)
        
        NewRegsMapChanged = (NewRegs != NewRegsMap[BB])
        NewRegsMap[BB] = NewRegs

        # NewRegsMap union RegsMap, prefer NewRegsMap at key conflict
        SuccessorRegsMap = {**RegsMap[BB], **NewRegsMap[BB]}  

        for SuccessorBB in BB._succs:
            if SuccessorBB in RegsMap:
                RegsMap[SuccessorBB].update(SuccessorRegsMap)
            else:
                RegsMap[SuccessorBB] = SuccessorRegsMap.copy()            

        return NewRegsMapChanged
    
    def ProcessBBRev(self, BB, NewRegsRevMap, RegsRevMap):
            
        NewRegs = {}

        for Inst in reversed(BB.instructions):

            if Inst.IsLoad() or Inst.IsAddrCompute():
                self.HandleLoadOrAddrCompute(Inst, RegsRevMap[BB], NewRegs)

            Inst.ResolveType()
            self.UpdateNewRegsMap(NewRegs, Inst)
        
        NewRegsMapChanged = (NewRegs != NewRegsRevMap[BB])
        NewRegsRevMap[BB] = NewRegs

        # NewRegsMap union RegsMap, prefer NewRegsMap at key conflict
        PredecessorRegsMap = {**RegsRevMap[BB], **NewRegsRevMap[BB]}  

        for PredecessorBB in BB._preds:
            if PredecessorBB in RegsRevMap:
                RegsRevMap[PredecessorBB].update(PredecessorRegsMap)
            else:
                RegsRevMap[PredecessorBB] = PredecessorRegsMap.copy()            

        return NewRegsMapChanged

    def UpdateNewRegsMap(self, NewRegs, Inst):

        for Operand in Inst._operands:
            
            if not Operand.IsReg:
                continue

            RegName = Operand.Reg

            if "NOTYPE" in Operand.TypeDesc:
                continue
            
            NewRegs[RegName] = Operand.TypeDesc

    def HandleStore(self, Inst, Regs, NewRegs):
        MergedRegs = {**Regs, **NewRegs}
        
        for i in range(1, len(Inst._operands)):
            UseOperand = Inst._operands[i]
            if UseOperand.IsReg and UseOperand.Reg in MergedRegs:
                Inst._operands[0]._TypeDesc = MergedRegs[UseOperand.Reg]+"_PTR"
                Inst._operands[1]._TypeDesc = MergedRegs[UseOperand.Reg]

        # for i in range(1, len(Inst._operands)):
        #     UseOperand = Inst._operands[i]
        #     if UseOperand.IsReg and UseOperand.Reg in MergedRegs:
        #         Inst._operands[i]._TypeDesc = MergedRegs[UseOperand.Reg]

    def HandleLoadOrAddrCompute(self, Inst, Regs, NewRegs):
        MergedRegs = {**Regs, **NewRegs}

        DefOperand = Inst._operands[0]

        if DefOperand.Reg in MergedRegs:
            Inst._operands[0]._TypeDesc = MergedRegs[DefOperand.Reg]

            if Inst.IsLoad():
                Inst._operands[1]._TypeDesc = MergedRegs[DefOperand.Reg]+"_PTR"
            if Inst.IsAddrCompute():
                Inst._operands[1]._TypeDesc = MergedRegs[DefOperand.Reg]


        # for i in range(1, len(Inst._operands)):
        #     UseOperand = Inst._operands[i]
        #     if UseOperand.IsReg and UseOperand.Reg in MergedRegs:
        #         Inst._operands[0]._TypeDesc = MergedRegs[UseOperand.Reg]
        #         return

    # Propagate type infromation
    def PropagateTypes(self, Func, Seeds):
        VisitedInsts = []
        while len(Seeds) > 0:
            Inst, BB = Seeds.pop(0)
            # Resolve type in instruction
            Inst.ResolveType()

            # Set visited
            VisitedInsts.append(Inst)
            
            # Propagate the uses for the value defined from this instruction
            Uses = self.PropagateUsesForDef(Inst, BB, Func, VisitedInsts)
            # Propagate the defs for the value used from this instruction
            Defs = self.PropagateDefForUses(Inst, BB, Func, VisitedInsts)

            # Add the uses into worklist
            for Use in Uses:
                if Use not in Seeds:
                    Seeds.append(Use)

            # Add the defs into worklist
            for Def in Defs:
                if Def not in Seeds:
                    Seeds.append(Def)

    # Collect the uses for the value defined in the given instruction
    def PropagateUsesForDef(self, Inst, BB, Func, VisitedInsts):
        Uses = []

        Def = Inst.GetDef()
        if Def == None:
            return Uses
            
        # Search uses in same basic block
        Insts = BB.instructions
        NeedCheck = False
        for i in range(len(Insts)):
            CurrInst = Insts[i]
            if NeedCheck:
                if CurrInst.IsStore():
                    # Check store instruction
                    if CurrInst.CheckAndUpdateUseType(Def) and not CurrInst in VisitedInsts:
                        Uses.append([CurrInst, BB])
                        VisitedInsts.append(CurrInst)
            else:
                if CurrInst == Inst:
                    NeedCheck = True

        return Uses
        
        
    # Collect the def for the values used in the given instruction
    def PropagateDefForUses(self, Inst, BB, Func, VisitedInsts):
        Defs = []

        Uses = Inst.GetUses()
    
        # Search defs in same basic block
        Insts = BB.instructions
        NeedCheck = False
        for i in reversed(range(len(Insts))):
            CurrInst = Insts[i]
            if NeedCheck:
                if CurrInst.IsAddrCompute():
                    if CurrInst.CheckAndUpdateDefType(Uses) and not CurrInst in VisitedInsts:
                        Defs.append([CurrInst, BB])
                        VisitedInsts.append(CurrInst)
                if CurrInst.IsLoad():
                    if CurrInst.CheckAndUpdateDefType(Uses) and not CurrInst in VisitedInsts:
                        Defs.append([CurrInst, BB])
                        VisitedInsts.append(CurrInst)
            else:
                if CurrInst == Inst:
                    NeedCheck = True
                        
        return Defs
