mutable struct KalmanSystemModel
    params::SCRIBEModelParameters
    estimates::Vector{LinearStateSystem}
    information::Vector{LinearInformationSystem}
end

struct LinearStateSystem
    state::SCRIBEModel
    observation::SCRIBEObserverState

    function LinearStateSystem(params::SCRIBEModelParameters, bhv::SCRIBEObserverBehavior)
        new(initialize_SCRIBEModel_from_parameters(params), bhv)
    end
end

struct LinearInformationSystem
    y::Vector{Float64}
    Y::Vector{Float64}
    i::Vector{Float64}
    I::Vector{Float64}
end

struct LinearSystemEstimators
    system::KalmanSystemModel
    A::Function
    ϕ::Function
    H::Function
    z::Function
end

