from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from lift.lifter import Lifter

ARG_OFFSET = 320 # 0x140

class Function:
    
    #def __init__(self, parser):
    #    self.name = ""
    #    self.ParseFunction(parser)

    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.args = []
        self.ArgMap = {}
        self.BlockMap = {}
        self.DefUse = None
        
    # Resovle instructions' operands
    def ResolveOperands(self, insts):
        self.args = []
        self.regs = []
        for inst in insts:
            args, regs = inst.GetArgsAndRegs()
            self.args.extend(args)
            self.regs.extend(regs)

    # Register argument with offset
    def RegisterArg(self, Offset, Arg):
        self.ArgMap[Offset] = Arg
        
    # Get the arguments for current function
    def GetArgs(self):
        ArgIdxes = []
        Args = []
        # Sort the map
        SortedArgs = {key: val for key, val in
                      sorted(self.ArgMap.items(), key = lambda ele: ele[0])}

        # Collect the keys
        for Offset, Operand in SortedArgs.items():
            if Offset < ARG_OFFSET:
                continue
            elif Operand.Skipped: # This is the case that the operand may be the associated operand
                continue
            else:
                ArgIdxes.append(Offset)
                Args.append(Operand)
                
        return ArgIdxes, Args

    # Get the registers used in this function
    def GetRegs(self, lifter):
        Regs = {}
        for BB in self.blocks:
            BB.GetRegs(Regs, lifter)

        return Regs
        
    # Create control flow graph
    def DumpCFG(self):
        for BB in self.blocks:
            succs = []
            for succ in BB.succs:
                succs.append(succ.addr_content)
            print("BB: ", BB.addr_content, succs)

    # Build the map between basic block and its IR version
    def BuildBBToIRMap(self, IRFunc, BlockMap):
        IsEntry = True
        for BB in self.blocks:
            if IsEntry:
                BBName = "EntryBB_" + BB.addr_content
            else:
                BBName = "BB_" + BB.addr_content

            # Create the basic block
            IRBlock = IRFunc.append_basic_block(BBName)
            # Register IR block
            BlockMap[BB] = IRBlock
            
            IsEntry = False
            
    # Lift to LLVM IR
    def Lift(self, lifter, llvm_module, func_name):
        ArgTypes = []
        # Collect arguments
        ArgIdxes, Args = self.GetArgs()
        # Get arg types
        for Arg in Args:
            ArgTypes.append(Arg.GetIRType(lifter))
        
        FuncTy = lifter.ir.FunctionType(lifter.ir.VoidType(), ArgTypes)
        IRFunc = lifter.ir.Function(llvm_module, FuncTy, self.name)

        # Construct the map based on IR basic block
        self.BuildBBToIRMap(IRFunc, self.BlockMap)
        
        IsEntry = True
        # The argument offset to IR argument map, that is created at the entry block code generation
        IRArgs = {}
        # The register name to IR register map, that is created at the entry block code generation
        IRRegs = {}
        
        for BB in self.blocks:
            # Get basic block
            IRBlock = self.BlockMap[BB] 
            # Create IR builder
            Builder = lifter.ir.IRBuilder(IRBlock)
            
            if IsEntry:
                ArgID = 0
                # Alloc the variable for arguments and update argument map
                for Arg in Args:
                    ArgName = "Arg" + str(ArgID)
                    IRArg = Builder.alloca(ArgTypes[ArgID], 8, ArgName)

                    # Register the IR argument
                    IRArgs[ArgIdxes[ArgID]] = IRArg
                    
                    # Store the argument values
                    Builder.store(IRFunc.args[ArgID], IRArg)

                    # Increment argument ID
                    ArgID = ArgID + 1
                    
                # Collect registers' name with type information
                Regs = self.GetRegs(lifter)

                # Alloc the variable for registers
                for Reg in Regs:
                    Operand = Regs[Reg]
                    RegName = Operand.GetIRRegName(lifter)
                    IRReg = Builder.alloca(Operand.GetIRType(lifter), 8, RegName)
                    # Register the IR registers
                    IRRegs[RegName] = IRReg


            # Lift the basic block content
            BB.Lift(lifter, Builder, IRRegs, IRArgs, self.BlockMap, IRFunc)

            IsEntry = False

        
