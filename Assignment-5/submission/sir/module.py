from sir.function import Function
from llvmlite import ir, binding

class Module :
    def __init__(self, name, parser):
        self.name = name
        #Please Donot change this varibale name (will be used for evaluation)
        self.branch_divergence_insts = [] 

        # Parse the module
        self.ParseModule(parser)

    def ParseModule(self, parser):
        self.functions = parser.apply()

    def Lift(self, lifter, outfile):
        # Generate module level information
        llvm_module = lifter.ir.Module(self.name)

        lifter.AddIntrinsics(llvm_module)
        
        for func in self.functions:
            # Lift function
            func.Lift(lifter, llvm_module, self.name)

        # print(llvm_module)
        
        # Generate IR file 
        print(llvm_module, file = outfile)
        
        
