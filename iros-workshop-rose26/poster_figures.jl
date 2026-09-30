include(joinpath(@__DIR__, "opportunistic_comms_distributed_exploration.jl"))

const POSTER_OUTPUT = joinpath(@__DIR__, "res", "paper")
const POSTER_COLORS = (:black, :royalblue, :firebrick, :purple, :seagreen, :darkorange)
const POSTER_LABELS = ("Centralized", "SCRIBE", "KF only", "CI only", "Last-2 observations", "Independent")

function poster_csv(path)
    lines = readlines(path)
    fields = Symbol.(split(first(lines), ','))
    map(lines[2:end]) do line
        values = map(zip(fields, split(line, ','))) do (field, value)
            field in (:backend, :topology, :reliability) ? Symbol(value) :
            field in (:left, :right) ? String(value) : parse(Float64, value)
        end
        Dict(zip(fields, values))
    end
end

function poster_axes!(panel; legend=:topright)
    plot!(panel; titlefontsize=24, guidefontsize=22, tickfontsize=18,
        legendfontsize=15, legend, legend_columns=2,
        left_margin=14Plots.mm, right_margin=9Plots.mm,
        bottom_margin=12Plots.mm, top_margin=8Plots.mm,
        gridalpha=0.12, aspect_ratio=:none)
end

function save_poster(panel, name)
    savefig(panel, joinpath(POSTER_OUTPUT, name * ".png"))
    println("Saved poster figure: $name")
end

function poster_asynchronous()
    output = asynchronous_output_directory(:full)
    environment = open(deserialize, joinpath(output, "plot_context.jls"))
    settings = asynchronous_fusion_settings(:full)
    trials = load_asynchronous_trials(settings, output, environment[:mission_steps])
    rows = reduce(vcat, getindex.(trials, :rows))
    trial = selected_asynchronous_trial(trials)
    history = trial[:agent_error_history]
    team = rmse_panel(rows; render_profile=:poster)
    agents = plot(; title="(b) Individual agents, n=$(trial[:n_agents])",
        xlabel="Elapsed timesteps", ylabel="Agent NRMSE")
    colors = [get(cgrad(:viridis), s) for s in range(0.08, 0.92; length=length(history[:ids]))]
    foreach(zip(history[:ids], colors)) do (aid, color)
        plot!(agents, history[:steps], history[:errors][aid]; color,
            linewidth=3, label=replace(aid, "agent" => "A"))
        steps = history[:communication_steps][aid]
        scatter!(agents, history[:steps][steps], history[:errors][aid][steps];
            markercolor=:white, markerstrokecolor=color, markerstrokewidth=1.4,
            markersize=4.2, label=false)
    end
    poster_axes!(team)
    poster_axes!(agents)
    plot!(team; legend=:top, legend_columns=4)
    plot!(agents; legend=:top, legend_columns=6,
        title="Agent RMSE, n=$(trial[:n_agents])",
        ylims=(ylims(agents)[1], 1.10ylims(agents)[2]),
        xlims=(0, 2000), xticks=0:500:2000)
    example = selected_reconstruction(environment, trial)
    limit = maximum(abs, environment[:truth])
    truth = spatial_panel(environment[:truth], environment, "(c) Ground truth", limit;
        render_profile=:poster, colorbar=true, left_margin=18Plots.mm,
        right_margin=7Plots.mm)
    posterior = spatial_panel(example[:field], environment, "(d) Final SCRIBE posterior", limit;
        render_profile=:poster, colorbar=true, show_latitude=false,
        left_margin=0Plots.mm, right_margin=12Plots.mm)
    aid = example[:agent]
    sampling_trajectory!(posterior, sampling_trajectory(environment, trial, aid),
        aid; render_profile=:poster)
    plot!(posterior; legend=false)
    nx, ny = environment[:roms][:grid_shape]
    note = "n=$(trial[:n_agents]), seed=$(trial[:seed])\n" *
        "agent NRMSE=$(round(example[:nrmse]; digits=3))\n" *
        "team NRMSE=$(round(trial[:final_metrics][:team_nrmse]; digits=3))\n" *
        "pink path: $aid samples"
    annotate!(posterior, 0.96nx, 0.07ny, text(note, 17, :right, :bottom, :white))
    foreach(panel -> plot!(panel; bottom_margin=18Plots.mm, top_margin=2Plots.mm),
        (truth, posterior))
    # Geographic panels retain their physical aspect; time-series panels do not.
    figure = plot(team, agents, truth, posterior;
        layout=@layout([grid(1, 2, widths=[0.44, 0.56]){0.56h};
            grid(1, 2, widths=[0.507, 0.493])]), size=(2600, 1320))
    save_poster(figure, "asynchronous_model_fusion_poster")
end

function poster_opportunistic()
    output = joinpath(@__DIR__, "res", "opportunistic_exploration", "full")
    rows = poster_csv(joinpath(output, "metric_history.csv"))
    contacts = poster_csv(joinpath(output, "contact_history.csv"))
    panels = map((:mean_agent_rmse, :disagreement),
        ("(a) Mean agent reconstruction", "(b) Maximum pairwise disagreement")) do metric, title
        panel = plot(; title, xlabel="Mission time (sampling intervals)", ylabel="RMSE")
        foreach(zip(OPPORTUNISTIC_BACKENDS, POSTER_LABELS, POSTER_COLORS)) do (backend, label, color)
            subset = filter(r -> r[:backend] == backend, rows)
            events = sort(unique(getindex.(subset, :event)))
            groups = [filter(r -> r[:event] == event, subset) for event in events]
            μ = [mean(getindex.(group, metric)) for group in groups]
            σ = [std(getindex.(group, metric); corrected=false) for group in groups]
            times = [mean(getindex.(group, :time)) for group in groups]
            plot!(panel, times, μ; color, ribbon=σ, fillalpha=0.12, label, linewidth=3)
            if backend ∉ (:centralized, :independent)
                encounters = Set(r[:event] for r in contacts if r[:backend] == backend)
                indices = findall(event -> event + 1 in encounters, events)
                scatter!(panel, times[indices], μ[indices]; markercolor=:white,
                    markerstrokecolor=color, markerstrokewidth=0.8,
                    markersize=2.3, label=false)
            end
        end
        scatter!(panel, [NaN], [NaN]; markercolor=:white,
            markerstrokecolor=:gray35, markersize=3, label="Before contact (≥1 trial)")
        poster_axes!(panel)
    end
    xmax = maximum(getindex.(rows, :time))
    foreach(panel -> plot!(panel; xlims=(0, xmax)), panels)
    figure = plot(panels...; layout=(1, 2), size=(2300, 800),
        bottom_margin=20Plots.mm, left_margin=16Plots.mm, right_margin=10Plots.mm)
    save_poster(figure, "opportunistic_performance")
end

function poster_scaling_panel(
    rows,
    metric,
    title,
    ylabel;
    render_profile=:paper,
    collapse_stable=false,
    centralized_reference=false,
    panel_position=:left,
)
    style = workshop_plot_style(render_profile)
    panel = plot(
        ;
        xlabel="Number of agents",
        ylabel,
        title,
        grid=true,
        legend=false,
        xlims=(1.7, 10.3),
        xticks=2:10,
        ylims=scaling_limits(
            rows,
            centralized_reference ?
                (metric, :centralized_nrmse) : metric;
            show_zero=metric in (
                :prediction_disagreement,
                :centralized_gap,
            ),
        ),
    )
    if collapse_stable
        stable = stable_scaling_aggregate(rows, metric)
        plot!(
            panel,
            getindex.(stable, :n_agents),
            getindex.(stable, :median);
            color=:black,
            linestyle=:solid,
            marker=:circle,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=false,
        )
        plotted = Vector{Vector{Float64}}()
        foreach((:sparse, :moderate, :dense)) do topology
            summary = scaling_aggregate(
                rows,
                topology,
                :intermittent,
                metric,
            )
            isempty(summary) && return
            values = Float64.(getindex.(summary, :median))
            duplicate = any(plotted) do previous
                length(previous) == length(values) && previous ≈ values
            end
            if !duplicate
                push!(plotted, values)
                plot!(
                    panel,
                    getindex.(summary, :n_agents),
                    values;
                    color=WORKSHOP_TOPOLOGY_COLORS[topology],
                    linestyle=:dash,
                    marker=:diamond,
                    linewidth=style.linewidth,
                    markersize=style.markersize,
                    label=false,
                )
            end
        end
    else
        foreach((:sparse, :moderate, :dense)) do topology
            foreach((:stable, :intermittent)) do reliability
                summary = scaling_aggregate(rows, topology, reliability, metric)
                if !isempty(summary)
                    plot!(
                        panel,
                        getindex.(summary, :n_agents),
                        getindex.(summary, :median);
                        color=WORKSHOP_TOPOLOGY_COLORS[topology],
                        linestyle=reliability == :stable ? :solid : :dash,
                        marker=reliability == :stable ? :circle : :diamond,
                        linewidth=style.linewidth,
                        markersize=style.markersize,
                        label=false,
                    )
                end
            end
        end
    end
    if centralized_reference
        reference = reference_scaling_aggregate(rows, :centralized_nrmse)
        plot!(
            panel,
            getindex.(reference, :n_agents),
            getindex.(reference, :median);
            color=:gray35,
            linestyle=:dot,
            marker=:utriangle,
            linewidth=style.linewidth,
            markersize=style.markersize,
            label=false,
        )
    end

    # Match labels to the original rendered series, retaining collapsed duplicates.
    foreach(panel.series_list) do series
        color = series[:seriescolor]
        series[:markersize] = 4
        series[:linewidth] = 3
        series[:label] = if series[:linestyle] == :dot
            "Centralized"
        elseif collapse_stable && series[:linestyle] == :solid
            "Stable (all topologies)"
        else
            topology = first(filter(t -> Plots.plot_color(WORKSHOP_TOPOLOGY_COLORS[t]) == color,
                (:sparse, :moderate, :dense)))
            "$(uppercasefirst(String(topology))) / $(series[:linestyle] == :solid ? "100%" : "50%")"
        end
    end
    poster_axes!(panel; legend=metric in (:fusion_seconds_per_agent_step,
        :consensus_iterations_per_agent_step, :payload_bytes_per_agent_step) ? :topleft : :topright)
    plot!(panel; legendfontsize=14)
end

function poster_scaling()
    rows = poster_csv(joinpath(scaling_output_directory(:full), "final_trial_metrics.csv"))
    metrics = (:mean_agent_nrmse, :worst_agent_nrmse, :prediction_disagreement, :centralized_gap)
    collapse = stable_topologies_coincide(rows, metrics)
    performance = map(enumerate(zip(metrics,
        ("(a) Mean-agent reconstruction", "(b) Worst-agent reconstruction",
         "(c) Inter-agent disagreement", "(d) Gap to centralized model")))) do (i, (metric, title))
        poster_scaling_panel(rows, metric, title, "NRMSE";
            render_profile=:poster, collapse_stable=collapse, centralized_reference=i == 1)
    end
    save_poster(plot(performance...; layout=(2, 2), size=(2400, 1500),
        bottom_margin=16Plots.mm), "agent_number_scaling_performance_poster")
    costs = map(zip(
        (:mean_integrated_variance, :fusion_seconds_per_agent_step,
         :consensus_iterations_per_agent_step, :payload_bytes_per_agent_step),
        ("(a) Posterior uncertainty", "(b) Fusion computation",
         "(c) Consensus effort", "(d) Communication payload"),
        ("Integrated variance", "Seconds / agent-update", "Rounds / agent-update", "Bytes / agent-update"))) do (metric, title, label)
        poster_scaling_panel(rows, metric, title, label; render_profile=:poster)
    end
    save_poster(plot(costs...; layout=(2, 2), size=(2400, 1500),
        bottom_margin=16Plots.mm), "agent_number_scaling_cost_poster")
end

function poster_snapshot()
    roms = open(deserialize, joinpath(WORKSHOP_EOF_DIRECTORY, "roms.jls"))
    params = load_eof_model_parameters(RAM_HEAD_EOF_MODEL)
    model = eof_model_at_coefficients(params, params.decomposition.coefficients[:, 3030])
    truth = roms[:data][:, 5387]
    rows = ram_head_sampling_rows(roms, 300)
    X = roms[:locations][rows, :]
    H = SCRIBE.eof_basis_at(model, X)
    z = truth[rows] - SCRIBE.eof_mean_at(model, X)
    variances = map(rows) do row
        only(eof_effective_measurement_covariance(model,
            permutedims(roms[:locations][row, :]), 1e-4))
    end
    prior = SCRIBE.init_agent_info(model.params)
    # Equivalent batch posterior for the original static, deterministic measurements.
    coefficients = Symmetric(prior.Y + H' * (H ./ variances)) \
        (prior.y + H' * (z ./ variances))
    posterior = reconstruct_eof_field(model; coefficients)
    rmse = sqrt(mean(abs2, posterior - truth))
    println("Snapshot 5387 recovered RMSE: $rmse; saved: 0.011848721836859118")
    isapprox(rmse, 0.011848721836859118; rtol=1e-7) ||
        error("Recovered snapshot does not match the saved comparison.")
    limit = maximum(abs, truth)
    environment = Dict(:roms => roms)
    panels = map((truth, posterior), ("ROMS snapshot 5387", "EOF posterior: 300 samples")) do field, title
        panel = spatial_panel(field, environment, title, limit; render_profile=:poster,
            left_margin=22Plots.mm, right_margin=4Plots.mm)
        plot!(panel; aspect_ratio=:equal, bottom_margin=20Plots.mm,
            top_margin=8Plots.mm, titlefontsize=24)
    end
    plot!(panels[2]; ylabel="", yticks=false, left_margin=4Plots.mm)
    levels = collect(range(-limit, limit; length=256))
    bar = heatmap([0, 1], levels, repeat(levels, 1, 2);
        color=:balance, clims=(-limit, limit), colorbar=false, xticks=false,
        ymirror=true, ylabel="Eastward velocity (m s⁻¹)", title=" ", xlabel=" ",
        titlefontsize=24, tickfontsize=18, guidefontsize=22,
        left_margin=3Plots.mm, right_margin=22Plots.mm,
        top_margin=12Plots.mm, bottom_margin=24Plots.mm)
    save_poster(plot(panels..., bar; layout=@layout([a{0.49w} b c{0.06w}]),
        size=(2400, 550)), "snapshot_5387")
end

"""Render saved workshop results into res/paper; no simulation or paper edits."""
function poster_figures()
    BLAS.set_num_threads(1)
    mkpath(POSTER_OUTPUT)
    foreach((poster_asynchronous, poster_opportunistic, poster_scaling, poster_snapshot)) do render
        render()
        GC.gc()
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && poster_figures()
