"""CI assimilation of measurement information, minimizing coefficient covariance trace."""
function ci_measurement_update(Y, y, δI, δi)
    iszero(δI) && return KFEnvInfo(copy(y), copy(Y), copy(δi), copy(δI))
    objective(ω) = tr(cholesky(Symmetric(ω * Y + (1 - ω) * δI)) \
        Matrix{eltype(Y)}(I, size(Y)...))
    # A positive prior weight keeps scalar-measurement updates nonsingular.
    left, right = sqrt(eps(Float64)), 1.0
    ratio = (sqrt(5) - 1) / 2
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    fa, fb = objective(a), objective(b)
    for _ in 1:64
        if fa < fb
            right, b, fb = b, a, fa
            a = right - ratio * (right - left)
            fa = objective(a)
        else
            left, a, fa = a, b, fb
            b = left + ratio * (right - left)
            fb = objective(b)
        end
    end
    ω = (left + right) / 2
    ω = objective(1.0) ≤ objective(ω) ? 1.0 : ω
    Y⁺ = ω * Y + (1 - ω) * δI
    KFEnvInfo(ω * y + (1 - ω) * δi, (Y⁺ + Y⁺') / 2,
        (1 - ω) * δi, (1 - ω) * δI)
end

function assimilate_shared_observation(info, observation)
    δ = measurement_information(observation[:H], observation[:z], observation[:R])
    KFEnvInfo(info.y + δ[:δi], info.Y + δ[:δI], δ[:δi], δ[:δI])
end
