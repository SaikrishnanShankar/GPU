from sir.operand import Operand
from sir.controlcode import ControlCode
from sir.controlcode import PresetCtlCodeException

class UnsupportedOperatorException(Exception):
    pass

class UnsupportedInstructionException(Exception):
    pass

class InvalidTypeException(Exception):
    pass

class Instruction:
    def __init__(self, id, opcodes, operands, inst_content):
        self._id = id
        self._opcodes = opcodes
        self._operands = operands
        self._InstContent = inst_content
        self._TwinIdx = ""
        self._TrueBranch = None
        self._FalseBranch = None
        self._CtlCode = None
        
    @property
    def id(self):
        return self._id
    
    @property
    def opcodes(self):
        return self._opcodes

    @property
    def operands(self):
        return self._operands

    @property
    def pflag(self):
        if self.IsPredicateReg(self._opcodes[0]):
            return self._opcodes[0]
        else:
            return None

    @property
    def controlcode(self):
        return self._CtlCode
    
    def GetArgsAndRegs(self):
        regs = []
        args = []
        for operand in self._operands:
            if operand.IsReg:
                regs.append(operand)
            if operand.IsArg:
                args.append(operand)

        return args, regs

    def SetCtlCode(self, CtlCode):
        if self._CtlCode != None:
            raise PresetCtlCodeException

        self._CtlCode = CtlCode
        
    def IsExit(self):
        if self._opcodes[0] == "EXIT":
            return True
        elif len(self._opcodes) > 1 and self._opcodes[1] == "EXIT":
            return True

        return False

    # def IsBranch(self):
    #     Idx = 0
    #     # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
    #     if self.IsPredicateReg(self._opcodes[Idx]):
    #         Idx = Idx + 1
    #     return self._opcodes[Idx] == "ISETP" 

    def IsBranch(self):
        if len(self.operands)>0 and self.GetDef().Reg:
            return self.IsPredicateReg(self.GetDef().Reg)

    def InCondPath(self):
        return self.IsPredicateReg(self._opcodes[0])

    def IsBinary(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "FFMA" or self._opcodes[Idx] == "FADD" or self._opcodes[Idx] == "XMAD" or self._opcodes[Idx] == "IMAD" or self._opcodes[Idx] == "SHL" or self._opcodes[Idx] == "SHR" or self._opcodes[Idx] == "SHF" or self._opcodes[Idx] == "S2R"
        # return self._opcodes[Idx] == "FFMA" or self._opcodes[Idx] == "FADD" or self._opcodes[Idx] == "XMAD" or self._opcodes[Idx] == "IMAD" or self._opcodes[Idx] == "SHL" or self._opcodes[Idx] == "SHR" or self._opcodes[Idx] == "SHF" or self._opcodes[Idx] == "S2R" or self._opcodes[Idx] == "ISCADD"
    
    
    def IsNOP(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "NOP"

    def IsAddrCompute(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        if self._opcodes[Idx] == "IADD":
            # Check operands
            if len(self._operands) == 3:
                operand = self._operands[2]
                # Check function argument operand
                return operand.IsArg

        return False

    def IsLoad(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "LDG" or self._opcodes[Idx] == "SULD"

    def IsStore(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        return self._opcodes[Idx] == "STG" or self._opcodes[Idx] == "SUST"

    # Set all operands as skipped
    def SetSkip(self):
        for Operand in self._operands:
            Operand.SetSkip()
            
    # def ResolveType(self):
    #     if not self.DirectlySolveType():
    #         if not self.PartialSolveType():
    #             print("checking ", self._opcodes[0])
    #             raise UnsupportedOperatorException

    def ResolveType(self):
        if self.DirectlySolveType():
            return True
        
        if self.PartialSolveType():
            return True
        
        return False

    # def ResolveType(self):
    #     if not self.DirectlySolveType():
    #         if not self.PartialSolveType():
    #             print("Failed: ",end="")
    #         else:
    #             print("Success: ",end="")
    #     else:
    #         print("Success: ",end="")
            
    #     print(self._InstContent+" => ",end="")
    #     for operand in self.operands:
    #         if operand.Name:
    #             print(operand.Name,end="")
    #         if operand.TypeDesc:
    #             print("("+operand.TypeDesc+"), ",end="")
    #     print("")

    # Collect registers used in instructions
    def GetRegs(self, Regs, lifter):
        # Check twin instruction case
        TwinIdx = self._TwinIdx
        
        for Operand in self._operands:
            if Operand.IsReg:
                if TwinIdx.find(Operand.Reg) == 0:
                    if not Operand.TypeDesc == "NOTYPE":
                        RegName = TwinIdx + Operand.TypeDesc
                        Regs[RegName] = Operand
                else:
                    if not Operand.TypeDesc == "NOTYPE":
                        Regs[Operand.GetIRRegName(lifter)] = Operand
        
    # Get def operand
    def GetDef(self):
        return self._operands[0]

    # Get use operand
    def GetUses(self):
        Uses = []
        for i in range(1, len(self._operands)):
            Uses.append(self._operands[i])

        return Uses

    # Get branch flag
    def GetBranchFlag(self):
        Operand = self._operands[0]
        if Operand.Name == "P0":
            return Operand.Name
        else:
            return None
        
    # Check and update the use operand's type from the givenn operand
    def CheckAndUpdateUseType(self, Def):
        for i in range(1, len(self._operands)):
            CurrOperand = self._operands[i]
            if CurrOperand.Name == Def.Name:                
                CurrOperand.SetTypeDesc(Def.TypeDesc)
                return True

        return False
    
    # Check and update the def operand's type from the given operands
    def CheckAndUpdateDefType(self, Uses):
        Def = self._operands[0]
        for i in range(len(Uses)):
            CurrUse = Uses[i]
            if CurrUse.IsReg and Def.IsReg and CurrUse.Reg == Def.Reg: # CurrUse.Name == Def.Name:
                Def.SetTypeDesc(CurrUse.TypeDesc)
                return True

        return False
    
    def IsPredicateReg(self, opcode):
        if opcode[0] == 'P' and opcode[1].isdigit():
            return True
        if opcode[0] == '!' and opcode[1] == 'P' and opcode[2].isdigit():
            return True
        return False
    
    # Directly resolve the type description, this is mainly working for binary operation
    def DirectlySolveType(self):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        #     Idx = Idx + 1
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
        TypeDesc = None
        if self._opcodes[Idx] == "FFMA":
            TypeDesc = "Float32"
        elif self._opcodes[Idx] == "FMUL":
            TypeDesc = "Float32"            
        elif self._opcodes[Idx] == "FADD":
            TypeDesc = "Float32"
        elif self._opcodes[Idx] == "XMAD":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "IMAD":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "ISCADD":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "SHL":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "SHR":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "SHF":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "S2R":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "ISUB":
            TypeDesc = "INT"
        elif self._opcodes[Idx] == "ISETP":
            TypeDesc = "INT"
        else:
            return False
        
        for operand in self._operands:
            operand.SetTypeDesc(TypeDesc)

        return True
    
    def PartialSolveType(self):
        if self._opcodes[0] == "LDG":
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[1].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[1].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == "SULD":
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[1].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[1].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == "STG":
            TypeDesc = self._operands[1].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[0].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[0].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == "SUST":
            TypeDesc = self._operands[1].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc and "PTR" not in TypeDesc:
                self._operands[0].SetTypeDesc(TypeDesc + "_PTR")
            else:
                TypeDesc = self._operands[0].GetTypeDesc()
                if TypeDesc != None and "NOTYPE" not in TypeDesc:
                    self._operands[0].SetTypeDesc(TypeDesc.replace('_PTR', ""))
                else:
                    return False
        elif self._opcodes[0] == 'IADD':
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc:
                self._operands[1].SetTypeDesc("Int32") # The integer offset
                self._operands[2].SetTypeDesc(TypeDesc)
        elif self._opcodes[0] == 'IADD32I':
            TypeDesc = self._operands[0].GetTypeDesc()
            if TypeDesc != None and "NOTYPE" not in TypeDesc:
                self._operands[1].SetTypeDesc("Int32") # The integer offset
                self._operands[2].SetTypeDesc(TypeDesc)
        else:
            return False

        return True

    def Lift(self, lifter, IRBuilder, IRRegs, IRArgs):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self._opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
            
        if self._opcodes[Idx] == "MOV":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
        
            if ResOp.IsReg and Op1.IsReg:
                try:
                    IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                    # Load value
                    #IRVal = IRBuilder.load(IROp1, "loadval")

                    # Store result
                    #IRBuilder.store(IRVal, IRRes)
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "MOV32I":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
        
            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

        
        elif self._opcodes[Idx] == "IMAD":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
        elif self._opcodes[Idx] == "XMAD":
            ResOp = self._operands[1]
            Op1 = self._operands[1]
        elif self._opcodes[Idx] == "EXIT":
            IRBuilder.ret_void()
        elif self._opcodes[Idx] == "FADD":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            if Res.IsReg and Op1.IsReg and Op2.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
            
                # Load operands
                Load1 = IRBuilder.load(IROp1, "load")
                Load2 = IRBuilder.load(IROp2, "load")

                # FADD instruction
                IRVal = IRBuilder.add(Load1, Load2, "fadd")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "ISCADD":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            if Res.IsReg and Op1.IsReg and Op2.IsReg:
                IRRes = IRRegs[Res.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
            
                # Load operands
                Load1 = IRBuilder.load(IROp1, "load")
                Load2 = IRBuilder.load(IROp2, "load")

                # FADD instruction
                IRVal = IRBuilder.add(Load1, Load2, "add")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "ISUB":
            Res = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]
            if Res.IsReg and Op1.IsReg and Op2.IsReg:
                try:
                    IRRes = IRRegs[Res.GetIRRegName(lifter)]
                    IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                    IROp2 = IRRegs[Op2.GetIRRegName(lifter)]
            
                    # Load operands
                    Load1 = IRBuilder.load(IROp1, "load")
                    Load2 = IRBuilder.load(IROp2, "load")

                    # FADD instruction
                    IRVal = IRBuilder.add(Load1, Load2, "sub")

                    # Store result
                    IRBuilder.store(IRVal, IRRes)
                except Exception as e:
                    lifter.lift_errors.append(e)
                    
        elif self._opcodes[Idx] == "SHL":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                # Load value
                Load1 = IRBuilder.load(IROp1, "loadval")

                # Add 0
                IRVal = IRBuilder.shl(Load1, lifter.ir.Constant(lifter.ir.IntType(32), 0), "add")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "SHR":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                # Load value
                Load1 = IRBuilder.load(IROp1, "loadval")

                # Add 0
                IRVal = IRBuilder.shl(Load1, lifter.ir.Constant(lifter.ir.IntType(32), 0), "add")

                # Store result
                IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "SHF":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if ResOp.IsReg and Op1.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]

                # Load value
                Load1 = IRBuilder.load(IROp1, "loadval")

                # Add 0
                IRVal = IRBuilder.shl(Load1, lifter.ir.Constant(lifter.ir.IntType(32), 0), "add")

                # Store result
                IRBuilder.store(IRVal, IRRes)
                
        elif self._opcodes[Idx] == "IADD":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg and ResOp.HasTypeDesc():
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)];
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRArgs[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")

                try:
                    # Add operands
                    IRVal = IRBuilder.gep(Load2, Indices, "addr")

                    # Store the value
                    IRBuilder.store(IRVal, IRRes)
                except Exception as e:
                    lifter.lift_errors.append(e)
            
        elif self._opcodes[Idx] == "IADD32I":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)];
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRArgs[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")
                
                # Add operands
                IRVal = IRBuilder.gep(Load2, Indices, "addr")

                # Store the value
                IRBuilder.store(IRVal, IRRes)
                
        elif self._opcodes[Idx] == "S2R":
            ResOp = self._operands[0]
            ValOp = self._operands[1]
            if ValOp.IsThreadIdx and ResOp.IsReg:
                IRResOp = IRRegs[ResOp.GetIRRegName(lifter)]
                
                # Call thread idx operation
                IRVal = IRBuilder.call(lifter.GetThreadIdx, [], "ThreadIdx")

                # Store the result
                IRBuilder.store(IRVal, IRResOp)
                
        elif self._opcodes[Idx] == "LDG":
            PtrOp = self._operands[1]
            ValOp = self._operands[0]
            if PtrOp.IsReg and ValOp.IsReg:
                IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                LoadPtr = IRBuilder.load(IRPtrOp, "loadptr")

                # Load instruction
                IRRes = IRBuilder.load(LoadPtr, "load_inst")

                # Store the result
                IRBuilder.store(IRRes, IRValOp)

        elif self._opcodes[Idx] == "SULD":
            PtrOp = self._operands[1]
            ValOp = self._operands[0]
            #if PtrOp.IsReg and ValOp.IsReg:
                #IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                #IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                #LoadPtr = IRBuilder.load(IRPtrOp, "loadptr")

                # Load instruction
                #IRRes = IRBuilder.load(LoadPtr, "load_inst")

                # Store the result
                #IRBuilder.store(IRRes, IRValOp)
                
        elif self._opcodes[Idx] == "STG":
            PtrOp = self._operands[0]
            ValOp = self._operands[1]
            if PtrOp.IsReg and ValOp.IsReg:
                IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                Addr = IRBuilder.load(IRPtrOp, "loadptr")
                Val = IRBuilder.load(IRValOp, "loadval")

                # Store instruction
                IRBuilder.store(Val, Addr)

        elif self._opcodes[Idx] == "SUST":
            PtrOp = self._operands[0]
            ValOp = self._operands[1]
            #if PtrOp.IsReg and ValOp.IsReg:
                #IRPtrOp = IRRegs[PtrOp.GetIRRegName(lifter)]
                #IRValOp = IRRegs[ValOp.GetIRRegName(lifter)]

                # Load operands
                #Addr = IRBuilder.load(IRPtrOp, "loadptr")
                #Val = IRBuilder.load(IRValOp, "loadval")

                # Store instruction
                # IRBuilder.store(Val, Addr)
        elif self._opcodes[Idx] == "FMUL":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsReg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)]
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRRegs[Op2.GetIRRegName(lifter)]

                # Load operands
                Load1 = IRBuilder.load(IROp1, "op1")
                Load2 = IRBuilder.load(IROp2, "op2")

                # Perform multiplication
                IRVal = IRBuilder.mul(Load1, Load2, "mul_result")

                # Store the result
                IRBuilder.store(IRVal, IRRes)
        elif self._opcodes[Idx] == "FFMA":
            ResOp = self._operands[0]
            Op1 = self._operands[1]
            Op2 = self._operands[2]

            if Op1.IsReg and Op2.IsArg:
                IRRes = IRRegs[ResOp.GetIRRegName(lifter)];
                IROp1 = IRRegs[Op1.GetIRRegName(lifter)]
                IROp2 = IRArgs[Op2.ArgOffset]

                # Load values
                Indices = []
                Load1 = IRBuilder.load(IROp1, "offset")
                Indices.append(Load1)
                Load2 = IRBuilder.load(IROp2, "ptr")
                
                # Add operands
                #IRVal = IRBuilder.mul(Load2, Indices, "mul")

                # Store the value
                #IRBuilder.store(IRVal, IRRes)

        elif self._opcodes[Idx] == "LD":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "LOP":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "LOP32I":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BFE":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BFI":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "ST":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "SSY":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BRA":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "BRK":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "IMNMX":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "PSETP":
            ResOp = self._opcodes[0]
        elif self._opcodes[Idx] == "PBK":
            ResOp = self._opcodes[0]
        # elif self._opcodes[Idx] == "@P1" or self._opcodes[Idx] == "@!P1":
        #     return
        elif self._opcodes[Idx] == "ISET":
            # Not implemented yet
            ResOp = self._operands[0]
        else:
            print("lift instruction: ", self._opcodes[Idx])
            raise UnsupportedInstructionException 
            
    # Lift branch instruction
    def LiftBranch(self, lifter, IRBuilder, IRRegs, IRArgs, TrueBr, FalseBr):
        Idx = 0
        # if self._opcodes[Idx] == "P0" or self.opcodes[Idx] == "!P0":
        if self.IsPredicateReg(self._opcodes[Idx]):
            Idx = Idx + 1
            
        if self._opcodes[Idx] == "ISETP":
            Val1Op = self._operands[2]
            Val2Op = self._operands[3]

            # Check register or arguments
            IRVal1 = None
            IRVal2 = None
            if Val1Op.IsReg:
                IRVal1 = IRRegs[Val1Op.GetIRRegName(lifter)]
            elif Val1Op.IsArg:
                IRVal1 = IRArgs[Val1Op.ArgOffset]

            if Val2Op.IsReg:
                IRVal2 = IRRegs[Val2Op.GetIRRegName(lifter)]
            elif Val2Op.IsArg:
                IRVal2 = IRArgs[Val2Op.ArgOffset]

            if not IRVal1 == None and not IRVal2 == None:
                Val1 = IRBuilder.load(IRVal1, "val1")
                Val2 = IRBuilder.load(IRVal2, "val2")

                # Calculate condition
                Cmp = IRBuilder.icmp_signed(lifter.GetCmpOp(self._opcodes[1]), Val1, Val2, "cmp")
                
                # Branch instruction
                IRBuilder.cbranch(Cmp, TrueBr, FalseBr)
            
    def dump(self):
        print("inst: ", self._id, self._opcodes)
        for operand in self._operands:
            operand.dump()
        
