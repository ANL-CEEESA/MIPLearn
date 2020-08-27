ENV["PYTHON"] = ARGS[1]

using Pkg
Pkg.instantiate()
Pkg.build("CPLEX")
Pkg.build("Gurobi")
Pkg.build("PyCall")

