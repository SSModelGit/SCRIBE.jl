include(joinpath(@__DIR__, "agent_number_scaling.jl"))

const OPPORTUNISTIC_BACKENDS = (
    :centralized, :scribe, :kf_only, :ci_only, :last_k_observations, :independent,
)

opportunistic_settings(profile) = merge(agent_scaling_settings(profile), Dict(
    :seeds => profile == :full ? Tuple(901:2:923) : (901,),
    :n_samples => profile == :full ? 240 : 8,
    :radius_fraction => 0.8,
    :last_k => 2,
))

function opportunistic_environment()
    load_asynchronous_environment()
end

function regional_problem(problem, region)
    rows = Set(filter(collect(keys(problem.neighbors))) do row
        i, j = problem.grid_locations[row, :]
        i in region.rows && j in region.columns
    end)
    neighbors = Dict(row => Dict(a => next for (a, next) in problem.neighbors[row]
        if next in rows) for row in rows)
    ROMSExplorationProblem(problem.sensor, neighbors, problem.locations,
        problem.grid_locations, problem.modes, problem.climatology,
        problem.measurement_variance, problem.variance_gram,
        problem.planning_rows, problem.initial_model, problem.steps)
end

function opportunistic_schedule(settings, seed)
    rng = MersenneTwister(seed)
    sort([(time=k * (0.85 + 0.1a) + 0.05rand(rng), agent="agent$a")
        for a in 1:4 for k in 1:settings[:n_samples]]; by=x -> x.time)
end


# Repeated in-range exchanges, with older links served first.
# Disjoint pairs keep every backend on the same pairwise fusion schedule.
function opportunistic_contacts(positions, adjacency, last_exchange, radius, event)
    candidates = sort(filter(adjacency) do (a, b)
        norm(positions[a] - positions[b]) ≤ radius
    end; by=pair -> (get(last_exchange, pair, 0), pair))
    contacts = Tuple{String, String}[]
    used = Set{String}()
    foreach(candidates) do (a, b)
        if a ∉ used && b ∉ used
            push!(contacts, (a, b))
            push!(used, a, b)
            last_exchange[(a, b)] = event
        end
    end
    edges = empty_edges(sort(collect(keys(positions))))
    foreach(contacts) do (a, b)
        push!(edges[a], b)
        push!(edges[b], a)
    end
    edges, contacts
end

function opportunistic_metrics(environment, information, central, seed, backend, event, time)
    fields = posterior_fields(environment[:model], information)
    ids = sort(collect(keys(fields)))
    rmse(x, y) = sqrt(mean(abs2, x - y))
    pooled = reconstruct_eof_field(environment[:model];
        coefficients=posterior_coefficient_moments(central)[:μ])
    (seed=seed, backend=backend, event=event, time=time,
        mean_agent_rmse=mean(rmse(field, environment[:truth]) for field in values(fields)),
        disagreement=maximum(rmse(fields[a], fields[b])
            for (i, a) in enumerate(ids) for b in ids[(i + 1):end]),
        pooled_rmse=rmse(pooled, environment[:truth]))
end

function opportunistic_update(backend, information, innovations, contacts, central)
    add(info, δ) = KFEnvInfo(info.y + δ.i, info.Y + δ.I, δ.i, δ.I)
    posterior = Dict(a => backend == :ci_only ?
        ci_measurement_update(info.Y, info.y, innovations[a].I, innovations[a].i) :
        add(info, innovations[a]) for (a, info) in information)
    @match backend begin
        :centralized => Dict(a => copy(central) for a in keys(information))
        :independent => posterior
        :kf_only => let
            foreach(contacts) do (a, b)
                posterior[a] = add(posterior[a], innovations[b])
                posterior[b] = add(posterior[b], innovations[a])
            end
            posterior
        end
        :ci_only => let
            foreach(contacts) do (a, b)
                infos = [posterior[a], posterior[b]]
                w = SCRIBE.covariance_intersection_weights(getproperty.(infos, :Y))
                Y, y = sum(w .* getproperty.(infos, :Y)), sum(w .* getproperty.(infos, :y))
                fused = KFEnvInfo(y, (Y + Y') / 2,
                    sum(w .* getproperty.(infos, :i)), sum(w .* getproperty.(infos, :I)))
                posterior[a], posterior[b] = copy(fused), copy(fused)
            end
            posterior
        end
    end
end

function share_recent_observations!(information, buffers, contacts)
    foreach(contacts) do (a, b)
        foreach(((a, b), (b, a))) do (sender, receiver)
            foreach(buffers[sender]) do observation
                information[receiver] = assimilate_shared_observation(
                    information[receiver], observation)
            end
        end
    end
    information
end

function run_opportunistic_trial(environment, settings, seed, backend)
    problem = scaling_problem(environment, settings)
    regions = operational_regions(environment[:roms][:grid_shape], 4)
    ids = agent_ids(4)
    problems = Dict(a => regional_problem(problem, r) for (a, r) in zip(ids, regions))
    radius = settings[:radius_fraction] * maximum(max(length(r.rows), length(r.columns)) for r in regions)
    rngs = Dict(a => MersenneTwister(seed + 100i) for (i, a) in enumerate(ids))
    states = Dict(a => ROMSExplorationState(rand(rngs[a], sort(collect(keys(problems[a].neighbors)))), Any[]) for a in ids)
    policies = Dict(a => scaling_policy(problems[a], settings, seed + 10_000i) for (i, a) in enumerate(ids))
    prior = problem.initial_model.information
    information = Dict(a => copy(prior) for a in ids)
    central = copy(prior)
    observations = Dict(a => Any[] for a in ids)
    network = backend == :scribe ? initialize_asynchronous_network(
        environment[:params], prior, observations, settings[:noise_variance]) : nothing
    buffers = Dict(a => Any[] for a in ids)
    last_exchange = Dict{Tuple{String, String}, Int}()
    adjacency = adjacent_agent_pairs(regions)
    rows, contacts_log, paths = Any[], Any[], Any[]
    push!(rows, opportunistic_metrics(environment, information, central, seed, backend, 0, 0.0))
    foreach(enumerate(opportunistic_schedule(settings, seed))) do (event, sample)
        a = sample.agent
        model = SCRIBEModelState(environment[:model], information[a], settings[:noise_variance])
        action = fixed_rollout_action(policies[a], states[a], model,
            settings[:n_samples] - length(states[a].history), settings[:planning_iterations])
        states[a] = gen(problems[a], states[a], action, rngs[a]).sp
        observation = observation_record(problem, states[a])
        positions = Dict(a => vec(problem.grid_locations[states[a].row, :]) for a in ids)
        edges, contacts = opportunistic_contacts(positions, adjacency, last_exchange, radius, event)
        foreach(contacts) do (left, right)
            push!(contacts_log, (seed=seed, backend=backend, event=event,
                time=sample.time, left=left, right=right))
        end
        push!(paths, (seed=seed, backend=backend, event=event, time=sample.time,
            agent=a, row=states[a].row, observation=observation[:raw_observation]))
        δ = measurement_information(observation[:H], observation[:z], observation[:R])
        innovation = (i=δ[:δi], I=δ[:δI])
        zero_innovation = (i=zeros(length(prior.y)), I=zeros(size(prior.Y)))
        innovations = Dict(b => b == a ? innovation : zero_innovation for b in ids)
        central = KFEnvInfo(central.y + innovation.i, central.Y + innovation.I, innovation.i, innovation.I)
        information = @match backend begin
            :scribe => let
                foreach(ids) do b
                    push!(observations[b], b == a ? observation : Dict(
                        :H => zeros(1, length(prior.y)), :z => [0.0], :R => ones(1, 1)))
                end
                asynchronous_network_step!(network, event, edges, settings)
                network_information(network)
            end
            :last_k_observations => let
                push!(buffers[a], observation)
                length(buffers[a]) > settings[:last_k] && popfirst!(buffers[a])
                local_info = opportunistic_update(:independent, information, innovations, contacts, central)
                share_recent_observations!(local_info, buffers, contacts)
            end
            _ => opportunistic_update(backend, information, innovations, contacts, central)
        end
        push!(rows, opportunistic_metrics(environment, information, central,
            seed, backend, event, sample.time))
    end
    (rows=rows, contacts=contacts_log, paths=paths, radius=radius,
        regions=regions, final_information=information)
end

function save_opportunistic_plots(rows, output; contacts=[])
    labels = ("Centralized", "SCRIBE", "KF only", "CI only", "Last-2 observations", "Independent")
    colors = (:black, :royalblue, :firebrick, :purple, :seagreen, :darkorange)
    panels = map((:mean_agent_rmse, :disagreement, :pooled_rmse),
        ("Mean agent reconstruction", "Maximum pairwise disagreement", "Pooled-data reconstruction")) do metric, title
        panel = plot(; xlabel="Mission time (sampling intervals)", ylabel="RMSE", title,
            legend=:topright, legend_columns=2, size=(1000, 640),
            titlefontsize=24, guidefontsize=22, tickfontsize=18, legendfontsize=14,
            left_margin=15Plots.mm, right_margin=8Plots.mm,
            bottom_margin=15Plots.mm, top_margin=8Plots.mm)
        foreach(zip(OPPORTUNISTIC_BACKENDS, labels, colors)) do (backend, label, color)
            subset = filter(r -> r.backend == backend, rows)
            events = sort(unique(getproperty.(subset, :event)))
            groups = [filter(r -> r.event == event, subset) for event in events]
            μ = [mean(getproperty.(group, metric)) for group in groups]
            σ = [std(getproperty.(group, metric); corrected=false) for group in groups]
            times = [mean(getproperty.(group, :time)) for group in groups]
            plot!(panel, times, μ; color, ribbon=σ, fillalpha=0.12, label, linewidth=3)
            # Contacts are logged even for the non-communicating reference methods.
            # Mark actual peer-sharing backends only, on their ensemble mean curves.
            if backend ∉ (:centralized, :independent)
                encounters = Set(r.event for r in contacts if r.backend == backend)
                indices = findall(event -> event in encounters, events)
                scatter!(panel, times[indices], μ[indices]; color,
                    markersize=1.6, markerstrokewidth=0, label=false)
            end
        end
        panel
    end
    foreach(zip(panels, ("reconstruction_rmse", "inter_agent_disagreement", "pooled_rmse"))) do (panel, name)
        savefig(panel, joinpath(output, name * ".png"))
    end
    savefig(plot(panels[1:2]...; layout=(1, 2), size=(2000, 700),
        left_margin=15Plots.mm, right_margin=8Plots.mm,
        bottom_margin=15Plots.mm, top_margin=8Plots.mm),
        joinpath(output, "opportunistic_performance.png"))
end

"""Replot saved metrics and contacts, without running any exploration trials."""
function plot_saved_opportunistic_exploration(; profile=:full)
    output = joinpath(@__DIR__, "res", "opportunistic_exploration", String(profile))
    rows = map(readlines(joinpath(output, "metric_history.csv"))[2:end]) do line
        seed, backend, event, time, rmse, disagreement, pooled = split(line, ',')
        (seed=parse(Int, seed), backend=Symbol(backend), event=parse(Int, event),
            time=parse(Float64, time), mean_agent_rmse=parse(Float64, rmse),
            disagreement=parse(Float64, disagreement), pooled_rmse=parse(Float64, pooled))
    end
    contacts = map(readlines(joinpath(output, "contact_history.csv"))[2:end]) do line
        seed, backend, event, time, left, right = split(line, ',')
        (seed=parse(Int, seed), backend=Symbol(backend), event=parse(Int, event),
            time=parse(Float64, time), left=left, right=right)
    end
    save_opportunistic_plots(rows, output; contacts)
end

function write_opportunistic_rows(rows, path, fields)
    open(path, "w") do io
        println(io, join(fields, ','))
        foreach(rows) do row
            println(io, join((getproperty(row, field) for field in fields), ','))
        end
    end
end

function opportunistic_main(profile=:smoke)
    BLAS.set_num_threads(1)
    settings = opportunistic_settings(profile)
    environment = opportunistic_environment()
    output = joinpath(@__DIR__, "res", "opportunistic_exploration", String(profile))
    mkpath(joinpath(output, "checkpoints"))
    rows, contacts, paths = Any[], Any[], Any[]
    foreach(settings[:seeds]) do seed
        foreach(OPPORTUNISTIC_BACKENDS) do backend
            trial = run_opportunistic_trial(environment, settings, seed, backend)
            open(joinpath(output, "checkpoints", "$(backend)_$(seed).jls"), "w") do io
                serialize(io, trial)
            end
            append!(rows, trial.rows)
            append!(contacts, trial.contacts)
            append!(paths, trial.paths)
            println("$backend seed=$seed: $(length(trial.contacts)) pair exchanges, radius=$(trial.radius)")
            flush(stdout)
        end
    end
    write_opportunistic_rows(rows, joinpath(output, "metric_history.csv"), keys(first(rows)))
    write_opportunistic_rows(contacts, joinpath(output, "contact_history.csv"), (:seed, :backend, :event, :time, :left, :right))
    write_opportunistic_rows(paths, joinpath(output, "observations.csv"), keys(first(paths)))
    save_opportunistic_plots(rows, output; contacts)
    rows
end

abspath(PROGRAM_FILE) == (@__FILE__) && opportunistic_main(isempty(ARGS) ? :smoke : Symbol(first(ARGS)))
