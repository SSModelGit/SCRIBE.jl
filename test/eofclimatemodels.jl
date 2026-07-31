using LinearAlgebra
using Random
using Statistics

struct SyntheticEOFSource
    data::Matrix{Float64}
end

import SCRIBE: eof_model_data_loader

function eof_model_data_loader(
    source::SyntheticEOFSource;
    scale::Real=1.0,
)
    scale .* source.data
end

function synthetic_eof_problem(; seed=42, n_features=18, n_samples=600)
    rng = MersenneTwister(seed)
    raw_modes = randn(rng, n_features, 3)
    modes = Matrix(qr(raw_modes).Q[:, 1:3])
    transition = [
        0.91 0.08 0.00
        -0.05 0.84 0.06
        0.00 -0.04 0.78
    ]
    coefficients = zeros(3, n_samples)
    coefficients[:, 1] .= randn(rng, 3)
    for k in 2:n_samples
        coefficients[:, k] .=
            transition * coefficients[:, k - 1] +
            0.08 .* randn(rng, 3)
    end
    field_mean = collect(range(1.0, 2.0; length=n_features))
    data = field_mean .+ modes * coefficients +
        0.002 .* randn(rng, n_features, n_samples)
    locations = hcat(
        collect(range(-1.0, 1.0; length=n_features)),
        collect(range(0.5, 2.0; length=n_features)),
    )
    weights = collect(range(0.5, 1.5; length=n_features))
    (
        data=data,
        locations=locations,
        weights=weights,
        transition=transition,
    )
end

@testset "EOF decomposition and dynamics" begin
    problem = synthetic_eof_problem()
    decomposition = fit_eof_decomposition(
        problem.data;
        weights=problem.weights,
        rank=3,
        algorithm=:exact,
    )

    @test size(decomposition.modes) == (18, 3)
    @test size(decomposition.coefficients) == (3, 600)
    @test transpose(decomposition.modes) *
          Diagonal(problem.weights) *
          decomposition.modes ≈ I atol=1e-10
    @test decomposition.explained_variance > 0.999
    @test sum(eof_variance_fraction(decomposition)) ≈
          decomposition.explained_variance
    @test eof_mode_grid(decomposition, 1, (3, 6)) ==
          reshape(eof_mode_values(decomposition, 1), 3, 6)

    reconstruction =
        decomposition.mean .+
        decomposition.modes * decomposition.coefficients
    anomaly = problem.data .- mean(problem.data; dims=2)
    @test norm(problem.data - reconstruction) / norm(anomaly) < 0.04

    dynamics = fit_eof_dynamics(
        decomposition.coefficients;
        ridge=1e-8,
    )
    previous = decomposition.coefficients[:, 1:(end - 1)]
    following = decomposition.coefficients[:, 2:end]
    @test size(dynamics.A) == (3, 3)
    @test isposdef(dynamics.Q)
    @test norm(following - dynamics.A * previous) <
          norm(following)
end

@testset "Randomized EOF decomposition" begin
    rng = MersenneTwister(7)
    data = randn(rng, 400, 4) * randn(rng, 4, 300)
    decomposition = fit_eof_decomposition(
        data;
        rank=4,
        algorithm=:randomized,
        oversample=4,
        power_iterations=1,
        rng,
    )
    @test decomposition.explained_variance > 1 - 1e-10
    @test transpose(decomposition.modes) *
          decomposition.modes ≈ I atol=1e-10
end

@testset "EOF model construction and loader dispatch" begin
    problem = synthetic_eof_problem()
    model = initialize_eof_climate_model(
        SyntheticEOFSource(problem.data);
        loader_kwargs=(scale=1.0,),
        locations=problem.locations,
        weights=problem.weights,
        rank=3,
        algorithm=:exact,
        interpolation=:nearest,
        metadata=Dict("source" => "synthetic"),
    )

    @test model isa EOFClimateModel
    @test model.params isa EOFClimateModelParameters
    @test model.params.metadata["source"] == "synthetic"
    @test get_model_time(model) == 1
    @test_throws ArgumentError eof_model_data_loader(nothing)

    coefficients = model.params.decomposition.coefficients[:, 20]
    state = EOFClimateModel(20, model.params, coefficients)
    field = reconstruct_eof_field(state)
    @test predict_SCRIBEModel(state, problem.locations) ≈ field
    @test predict_SCRIBEModel(state, vec(problem.locations[3, :])) ≈ field[3]

    H, returned_locations =
        compute_obs_dynamics(state, problem.locations[1:4, :])
    @test returned_locations == problem.locations[1:4, :]
    @test H ≈ model.params.decomposition.modes[1:4, :]

    advanced = update_SCRIBEModel(state, coefficients .+ 0.1)
    @test advanced.k == 21
    @test advanced.ϕ ≈ coefficients .+ 0.1
    @test update_SCRIBEModel(state) isa EOFClimateModel
end

@testset "EOF observation and information-filter dispatch" begin
    problem = synthetic_eof_problem()
    params = EOFClimateModelParameters(
        problem.data;
        locations=problem.locations,
        weights=problem.weights,
        rank=3,
        algorithm=:exact,
        interpolation=:nearest,
    )
    coefficients = params.decomposition.coefficients[:, 25]
    world = EOFClimateModel(1, params, coefficients)
    X = problem.locations[1:8, :]
    behavior = EOFObserverBehavior(
        1e-4;
        include_truncation_error=true,
    )
    @test measurement_noise_covariance(behavior, size(X, 1)) ≈
          1e-4 .* Matrix{Float64}(I, size(X, 1), size(X, 1))
    observation = scribe_observations(X, world, behavior)

    @test observation isa EOFObserverState
    @test observation.mean ≈ params.decomposition.mean[1:8]
    @test observation.H ≈ params.decomposition.modes[1:8, :]
    @test observation.v[:R] - observation.v[:R_sensor] ≈
          Diagonal(params.decomposition.residual_variance[1:8])

    estimators = initialize_KF(params, behavior, X, world)
    @test estimators isa KFEstimators
    @test posterior_coefficient_moments(
        estimators.system.information[1],
    ).Σ ≈ params.P₀

    innovation = compute_innov_from_obs(estimators, 1)
    observed = estimators.system.estimates[1].observations
    expected_anomaly = observed.z - observed.mean
    expected = measurement_information(
        observed.H,
        expected_anomaly,
        observed.v[:R],
    )
    @test innovation[1] ≈ expected.δI
    @test innovation[2] ≈ expected.δi

    prior = estimators.system.information[1]
    exact_z = eof_mean_at(world, X) + eof_basis_at(world, X) * coefficients
    posterior = condition_on_measurement(world, prior, X, exact_z, 1e-8)
    posterior_coefficients = posterior_coefficient_moments(posterior)
    @test norm(posterior_coefficients.μ - coefficients) <
          norm(coefficients)

    moments = posterior_model_moments(world, posterior, X)
    @test length(moments.μ) == size(X, 1)
    @test isposdef(moments.Σ)

    fused = only(centralized_fusion([estimators], 1))
    @test fused isa KFEnvInfo
    progress_agent_env_filter(
        estimators.system,
        fused,
        update_SCRIBEModel(world),
        X,
    )
    @test estimators.system.k == 2
    @test estimators.system.estimates[2].estimate isa EOFClimateModel
end

@testset "EOF MATLAB artifact round trip" begin
    problem = synthetic_eof_problem(n_samples=120)
    model = initialize_eof_climate_model(
        problem.data;
        locations=problem.locations,
        weights=problem.weights,
        rank=3,
        algorithm=:exact,
        interpolation=:nearest,
        metadata=Dict(
            "source" => "round-trip",
            "sample_interval_hours" => 1.0,
        ),
    )

    mktempdir() do directory
        artifact_path = joinpath(directory, "learned_eof.mat")
        @test save_eof_model(artifact_path, model) == artifact_path
        loaded = load_eof_climate_model(artifact_path; k=9)
        @test loaded.k == 9
        @test loaded.params.decomposition.mean ≈
              model.params.decomposition.mean
        @test loaded.params.decomposition.modes ≈
              model.params.decomposition.modes
        @test loaded.params.decomposition.coefficients ≈
              model.params.decomposition.coefficients
        @test loaded.params.A ≈ model.params.A
        @test loaded.params.Q ≈ model.params.Q
        @test loaded.params.P₀ ≈ model.params.P₀
        @test loaded.params.locations ≈ model.params.locations
        @test loaded.params.interpolation == :nearest
        @test loaded.params.metadata["source"] == "round-trip"

        compact_path = joinpath(directory, "compact_eof.mat")
        save_eof_model(
            compact_path,
            model;
            include_coefficients=false,
        )
        compact = load_eof_climate_model(compact_path)
        @test isempty(compact.params.decomposition.coefficients)
        @test compact.params.A ≈ model.params.A
    end
end
