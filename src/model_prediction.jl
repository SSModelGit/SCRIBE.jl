using Match: @match
using Statistics: mean

import .SCRIBEModels: predict_SCRIBEModel

export prediction_dynamics, measurement_noise_covariance
export posterior_coefficient_moments, posterior_model_moments
export posterior_measurement_moments, posterior_measurement_distribution
export recover_covariance_from_info, predict_model_uncertainty
export predict_measurement_uncertainty, evaluate_information_metric
export measurement_information, condition_on_measurement
export logdet_positive_definite, D_KL, mutual_information

"""Evaluate the mean of a SCRIBE model at a location."""
function predict_SCRIBEModel(
    smodel::LGSFModel,
    x,
)
    let H=prediction_dynamics(smodel, x)
        only(H * smodel.ϕ)
    end
end

"""Convert one or more prediction locations into a row-oriented matrix."""
prediction_locations(x::Number) = reshape([x], 1, 1)
prediction_locations(x::AbstractVector) = reshape(x, 1, :)
prediction_locations(X) = X

"""Construct the model dynamics matrix associated with prediction locations.

Each row of the returned matrix is `ψ(x)'` for one row-location in `X`.
"""
function prediction_dynamics(smodel::SCRIBEModel, X)
    let Xₚ=prediction_locations(X),
        H=compute_obs_dynamics(smodel, Xₚ)[1]
        H
    end
end

function prediction_dynamics(smodel::LGSFModel, X)
    let Xₚ=prediction_locations(X)
        mapslices(smodel.ψ, Xₚ; dims=2)
    end
end

"""Return a numerically symmetric matrix."""
prediction_symmetric(M) = (M + M') / 2

"""Construct a measurement-noise covariance matrix.

Scalar inputs are interpreted as variances, while vector inputs contain one
variance per measurement.
"""
measurement_noise_covariance(σ²::Number, nₛ) = σ² * I(nₛ)

measurement_noise_covariance(σ²::AbstractVector, _) = Diagonal(σ²)

measurement_noise_covariance(R::UniformScaling, nₛ) =
    measurement_noise_covariance(R.λ, nₛ)

measurement_noise_covariance(R::AbstractMatrix, _) =
    prediction_symmetric(R)

function measurement_noise_covariance(observer::LGSFObserverBehavior,
                                      nₛ)
    measurement_noise_covariance(observer.v_s[:σ], nₛ)
end

"""Return the Cholesky factorization of a positive-definite matrix."""
factor_positive_definite(M) =
    cholesky(Symmetric(prediction_symmetric(M)); check=false)

"""Recover the coefficient covariance represented by an information state."""
function recover_covariance_from_info(info::KFEnvInfo)
    let Y_factor=factor_positive_definite(info.Y)
        prediction_symmetric(inv(Y_factor))
    end
end

"""Recover the posterior mean and covariance of the model coefficients."""
function posterior_coefficient_moments(info::KFEnvInfo)
    let Σ=recover_covariance_from_info(info),
        μ=Σ * info.y
        (μ=μ, Σ=Σ)
    end
end

"""Predict the posterior moments of the latent scalar field."""
function posterior_model_moments(smodel::SCRIBEModel, info::KFEnvInfo, X)
    let H=prediction_dynamics(smodel, X),
        coefficients=posterior_coefficient_moments(info)
        μ=H * coefficients.μ
        Σ=H * coefficients.Σ * H'
        (μ=μ, Σ=prediction_symmetric(Σ))
    end
end

"""Predict the posterior moments of noisy measurements at `X`."""
function posterior_measurement_moments(smodel::SCRIBEModel, info::KFEnvInfo,
                                       X, R)
    let latent=posterior_model_moments(smodel, info, X),
        Rₛ=measurement_noise_covariance(R, length(latent.μ))
        (μ=latent.μ, Σ=prediction_symmetric(latent.Σ + Rₛ))
    end
end

"""Return the Gaussian posterior predictive distribution of measurements."""
function posterior_measurement_distribution(smodel::SCRIBEModel,
                                            info::KFEnvInfo, X, R)
    let moments=posterior_measurement_moments(smodel, info, X, R)
        Gaussian(moments.μ, moments.Σ)
    end
end

"""Evaluate a covariance according to the requested uncertainty metric."""
function evaluate_prediction_covariance(Σ, metric::Symbol)
    let Σₛ=prediction_symmetric(Σ),
        σ²=diag(Σₛ)
        @match metric begin
            :covariance => Σₛ
            :variance => σ²
            :standard_deviation => sqrt.(σ²)
            :total_variance => sum(σ²)
            :mean_variance => mean(σ²)
            _ => throw(ArgumentError("Unknown uncertainty metric: $metric"))
        end
    end
end

function single_location_uncertainty(uncertainty, metric::Symbol)
    @match metric begin
        :variance => only(uncertainty)
        :standard_deviation => only(uncertainty)
        _ => uncertainty
    end
end

"""Predict latent-field uncertainty at one or more locations."""
function predict_model_uncertainty(smodel::SCRIBEModel, info::KFEnvInfo, X;
                                   metric::Symbol=:variance)
    let moments=posterior_model_moments(smodel, info, X),
        uncertainty=evaluate_prediction_covariance(moments.Σ, metric)
        X isa AbstractMatrix ?
            uncertainty :
            single_location_uncertainty(uncertainty, metric)
    end
end

"""Predict noisy-measurement uncertainty at one or more locations."""
function predict_measurement_uncertainty(smodel::SCRIBEModel,
                                         info::KFEnvInfo, X, R;
                                         metric::Symbol=:variance)
    let moments=posterior_measurement_moments(smodel, info, X, R),
        uncertainty=evaluate_prediction_covariance(moments.Σ, metric)
        X isa AbstractMatrix ?
            uncertainty :
            single_location_uncertainty(uncertainty, metric)
    end
end

"""Compute a scalar or matrix uncertainty metric from an information state."""
function evaluate_information_metric(info::KFEnvInfo;
                                     metric::Symbol=:differential_entropy)
    let Y=prediction_symmetric(info.Y),
        nᵩ=size(Y, 1)
        @match metric begin
            :information => Y
            :covariance => recover_covariance_from_info(info)
            :logdet_information => logdet_positive_definite(Y)
            :differential_entropy =>
                0.5 * (nᵩ * log(2π * exp(1)) -
                       logdet_positive_definite(Y))
            :total_variance => tr(recover_covariance_from_info(info))
            :mean_variance =>
                tr(recover_covariance_from_info(info)) / nᵩ
            :minimum_information_eigenvalue => eigmin(Symmetric(Y))
            :condition_number => cond(Y)
            _ => throw(ArgumentError("Unknown information metric: $metric"))
        end
    end
end

"""Compute the information-matrix innovation `δI = H'R⁻¹H`."""
measurement_information(H, R) =
    measurement_information(H, nothing, R)

"""Compute measurement innovations `(δI, δi)` for observed values `z`."""
function measurement_information(H, z, R)
    let Rₛ=measurement_noise_covariance(R, size(H, 1)),
        R_factor=factor_positive_definite(Rₛ),
        δI=prediction_symmetric(H' * (R_factor \ H))
        if isnothing(z)
            δI
        else
            let zₛ=z isa Number ? [z] : z,
                δi=H' * (R_factor \ zₛ)
                (δI=δI, δi=δi)
            end
        end
    end
end

"""Condition an information state on hypothetical or realized measurements.

The supplied `info` should be the prior at the time represented by `z`.
"""
function condition_on_measurement(smodel::SCRIBEModel, info::KFEnvInfo,
                                  X, z, R)
    let H=prediction_dynamics(smodel, X),
        innovation=measurement_information(H, z, R),
        Y⁺=prediction_symmetric(info.Y + innovation.δI),
        y⁺=info.y + innovation.δi
        KFEnvInfo(y⁺, Y⁺, innovation.δi, innovation.δI)
    end
end

"""Compute a stable log-determinant for a positive-definite matrix."""
function logdet_positive_definite(
    M_factor::LinearAlgebra.Cholesky,
)
    2 * sum(log, diag(M_factor.L))
end

logdet_positive_definite(M) =
    logdet_positive_definite(factor_positive_definite(M))

"""Evaluate `D_KL(p || q)` between multivariate Gaussian distributions."""
function D_KL(μₚ, Σₚ, μ_q, Σ_q)
    let Σₚ=prediction_symmetric(Σₚ),
        Σ_q=prediction_symmetric(Σ_q),
        nᵩ=length(μₚ)
        p_factor=factor_positive_definite(Σₚ)
        q_factor=factor_positive_definite(Σ_q)
        Δμ=μ_q - μₚ
        trace_term=tr(q_factor \ Σₚ)
        mean_term=Δμ ⋅ (q_factor \ Δμ)
        logdet_term=
            logdet_positive_definite(q_factor) -
            logdet_positive_definite(p_factor)
        0.5 * (trace_term + mean_term - nᵩ + logdet_term)
    end
end

"""Evaluate coefficient-posterior `D_KL(p || q)`.

Both information states must use the same basis functions in the same order.
"""
function D_KL(p::KFEnvInfo, q::KFEnvInfo)
    let p_moments=posterior_coefficient_moments(p),
        q_moments=posterior_coefficient_moments(q)
        D_KL(p_moments.μ, p_moments.Σ,
             q_moments.μ, q_moments.Σ)
    end
end

"""Evaluate realized information gain from conditioning on `z` at `X`."""
function D_KL(smodel::SCRIBEModel, prior::KFEnvInfo, X, z, R)
    let posterior=condition_on_measurement(smodel, prior, X, z, R)
        D_KL(posterior, prior)
    end
end

function information_units(Iₙ, units::Symbol)
    @match units begin
        :nats => Iₙ
        :bits => Iₙ / log(2)
        _ => throw(ArgumentError("Unknown information units: $units"))
    end
end

"""Evaluate expected mutual information for measurement dynamics `H`."""
function mutual_information(info::KFEnvInfo, H, R;
                            units::Symbol=:nats)
    let δI=measurement_information(H, R),
        Y=prediction_symmetric(info.Y),
        Y⁺=prediction_symmetric(Y + δI),
        Iₙ=0.5 * (
            logdet_positive_definite(Y⁺) -
            logdet_positive_definite(Y)
        )
        information_units(Iₙ, units)
    end
end

"""Evaluate expected mutual information conditioned on sampling at `X`."""
function mutual_information(smodel::SCRIBEModel, info::KFEnvInfo, X, R;
                            units::Symbol=:nats)
    let H=prediction_dynamics(smodel, X)
        mutual_information(info, H, R; units)
    end
end
