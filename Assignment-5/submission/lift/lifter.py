from llvmlite import ir, binding

class Lifter :
    def __init__(self):
        # Initialize LLVM environment
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        self.ir = ir
        self.lift_errors = []
        
    def GetCmpOp(self, Opcode):
        if Opcode == "GE":
            return ">="
        elif Opcode == "EQ":
            return "=="
        elif Opcode == "LE":
            return "<="
        elif Opcode == "GT":
            return ">"
        elif Opcode == "LT":
            return "<"

        return ""
        
            
    def AddIntrinsics(self, llvm_module):
        # Create thread idx function
        FuncTy = self.ir.FunctionType(self.ir.IntType(32), [])
        FuncName = "thread_idx"
        IRFunc = self.ir.Function(llvm_module, FuncTy, FuncName)

        self.GetThreadIdx = IRFunc
        
    def LiftModule(self, module, file):
        module.lift(self, file)

    def GetIRType(self, TypeDesc):
        if TypeDesc == "Int32":
            return self.ir.IntType(32)
        elif TypeDesc == "Float32":
            return self.ir.FloatType()
        elif TypeDesc == "Int32_PTR":
            return self.ir.PointerType(self.ir.IntType(32))
        elif TypeDesc == "Float32_PTR":
            return self.ir.PointerType(self.ir.FloatType())

        return self.ir.IntType(32)

    def Shutdown(self):
        # Cleanup LLVM environment
        binding.shutdown()
