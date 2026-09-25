using LinearAlgebra

@testset "offline EOF uncertainty calibration" begin
    coefficients = [
        -1.0 -0.4 0.2 0.8 1.1
         0.6  0.2 0.0 0.1 0.5
    ]
    decomposition = EOFDecomposition(
        zeros(3),
        [1.0 0.0; 0.0 1.0; 0.0 0.0],
        [1.0, 0.4],
        coefficients,
        ones(3),
        zeros(3),
        0.9,
        1.4,
        size(coefficients, 2),
    )
    calibration = calibrate_eof_uncertainty(decomposition)
    @test size(calibration.P₀) == (2, 2)
    @test size(calibration.Q) == (2, 2)
    @test isposdef(calibration.P₀)
    @test isposdef(calibration.Q)
    @test calibration.transition_samples == 4

    histories = [coefficients[:, 1:3], coefficients[:, 4:5]]
    bounded = calibrate_eof_uncertainty(decomposition; histories)
    @test bounded.transition_samples == 3

    initial_errors = [0.2 -0.1 0.0; -0.2 0.1 0.3]
    innovations = [0.02 -0.01 0.03; 0.01 0.04 -0.02]
    held_out = calibrate_eof_uncertainty(
        decomposition;
        initial_errors,
        process_innovations=innovations,
    )
    @test held_out.initial_samples == 3
    @test held_out.transition_samples == 3

    params = EOFClimateModelParameters(
        decomposition;
        process_covariance=Matrix{Float64}(I, 2, 2),
    )
    calibrated = with_eof_uncertainty(params, calibration)
    @test calibrated.P₀ == calibration.P₀
    @test calibrated.Q == calibration.Q
end
