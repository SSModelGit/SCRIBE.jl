using NCDatasets: NCDataset

export SOMModel

struct SOMModel <: SCRIBEModel
    data::Dict{Symbol, Any}
end

function SOMModel(path::String)
    NCDataset(path, "r") do ds
        vertex_names = sort(filter(k->startswith(k, "vertex_0"), collect(keys(ds))))
        SOMModel(Dict{Symbol, Any}(
            :x => Float64.(ds["x_coords"][:]),
            :y => Float64.(ds["y_coords"][:]),
            :components => ds["component"][:],
            :connectivity => Float64.(ds["topology_connectivity"][:,:]),
            :mask => Bool.(permutedims(ds["mask"][:,:,:], (3,2,1))),
            :vertex_ids => Int.(ds["vertex_id"][:]),
            :vertex_names => Symbol.(vertex_names),
            :vertices => [permutedims(coalesce.(ds[k][:,:,:], NaN), (3,2,1))
                for k in vertex_names
            ],
        ))
    end
end