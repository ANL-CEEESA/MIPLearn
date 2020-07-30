using PackageCompiler

using CPLEX
using CPLEXW
using Gurobi
using JuMP
using MathOptInterface
using PyCall
using TimerOutputs
using TinyBnB

pkg = [:CPLEX
       :CPLEXW
       :Gurobi
       :JuMP
       :MathOptInterface
       :PyCall
       :TimerOutputs
       :TinyBnB]

@info "Building system image..."
create_sysimage(pkg, sysimage_path="build/sysimage.so")