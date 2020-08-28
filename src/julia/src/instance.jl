#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JSON2
import Base: dump

get_instance_features(instance) = [0.]
get_variable_features(instance, var, index) = [0.]
find_violated_lazy_constraints(instance, model) = []
build_lazy_constraint(instance, model, v) = nothing

dump(instance::PyCall.PyObject, filename) = @pycall instance.dump(filename)
load!(instance::PyCall.PyObject, filename) = @pycall instance.load(filename)

macro Instance(klass)
    quote
        @pydef mutable struct Wrapper <: Instance
            function __init__(self, args...; kwargs...)
                self.data = $(esc(klass))(args...; kwargs...)
            end
                
            function dump(self, filename)
                prev_data = self.data
                self.data = JSON2.write(prev_data)
                Instance.dump(self, filename)
                self.data = prev_data
            end
            
            function load(self, filename)
                Instance.load(self, filename)
                self.data = JSON2.read(self.data, $(esc(klass)))
            end
                
            to_model(self) =
                $(esc(:to_model))(self.data)
                
            get_instance_features(self) =
                get_instance_features(self.data)
                
            get_variable_features(self, var, index) =
                get_variable_features(self.data, var, index)

            function find_violated_lazy_constraints(self, model)
                find_violated_lazy_constraints(self.data, model)
            end
            
            function build_lazy_constraint(self, model, v)
                build_lazy_constraint(self.data, model, v)
            end
        end
    end
end

export get_instance_features,
       get_variable_features,
       find_violated_lazy_constraints,
       build_lazy_constraint,
       dump,
       load!,
       @Instance