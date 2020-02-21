# MIPLearn, an extensible framework for Learning-Enhanced Mixed-Integer Optimization
# Copyright © 2020, UChicago Argonne, LLC. All rights reserved.
# Released under the modified BSD license. See COPYING.md for more details.
# Written by Alinson S. Xavier <axavier@anl.gov>

import Base.Threads.@threads
using TinyBnB, CPLEXW, Printf

instance_name = ARGS[1]
output_filename = ARGS[2]
node_limit = parse(Int, ARGS[3])

mip = open_mip(instance_name)
n_vars = CPXgetnumcols(mip.cplex_env[1], mip.cplex_lp[1])

pseudocost_count_up = [0 for i in 1:n_vars]
pseudocost_count_down = [0 for i in 1:n_vars]
pseudocost_sum_up = [0. for i in 1:n_vars]
pseudocost_sum_down = [0. for i in 1:n_vars]

function full_strong_branching_track(node::Node, progress::Progress)::TinyBnB.Variable
    N = length(node.fractional_variables)
    scores = Array{Float64}(undef, N)
    rates_up = Array{Float64}(undef, N) 
    rates_down = Array{Float64}(undef, N)
    
    @threads for v in 1:N
        fix_vars!(node.mip, node.branch_variables, node.branch_values)
        obj_up, obj_down = TinyBnB.probe(node.mip, node.fractional_variables[v])
        unfix_vars!(node.mip, node.branch_variables)
        delta_up  = obj_up - node.obj
        delta_down = obj_down - node.obj
        frac_up = ceil(node.fractional_values[v]) - node.fractional_values[v]
        frac_down = node.fractional_values[v] - floor(node.fractional_values[v])
        rates_up[v] = delta_up / frac_up
        rates_down[v] = delta_down / frac_down
        scores[v] = delta_up * delta_down
    end

    max_score, max_offset = findmax(scores)
    selected_var = node.fractional_variables[max_offset]
    
    if abs(rates_up[max_offset]) < 1e6
        pseudocost_count_up[selected_var.index] += 1
        pseudocost_sum_up[selected_var.index] += rates_up[max_offset]
    end
    
    if abs(rates_down[max_offset]) < 1e6
        pseudocost_count_down[selected_var.index] += 1
        pseudocost_sum_down[selected_var.index] += rates_down[max_offset]
    end
    
    return selected_var
end

branch_and_bound(mip,
                 node_limit = node_limit,
                 branch_rule = full_strong_branching_track,
                 node_rule = best_bound,
                 print_interval = 100)

priority = [(pseudocost_count_up[v] == 0 || pseudocost_count_down[v] == 0) ? 0 :
              (pseudocost_sum_up[v] / pseudocost_count_up[v]) *
              (pseudocost_sum_down[v] / pseudocost_count_down[v])
             for v in 1:n_vars];

open(output_filename, "w") do file
    for var in mip.binary_variables
        write(file, @sprintf("%s,%.0f\n", name(mip, var), priority[var.index]))
    end
end
