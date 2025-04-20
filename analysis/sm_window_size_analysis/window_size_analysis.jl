#=
This file  loads the `data_hep*.csv` CSVs in `/data` and pulls data from the `InteroceptionPrimals` repo.
It then removes any missing/NA data, prints summary statistics on the data, and then performs the Window Size Analysis and the Epoch Analysis on them.
The results can be found in `tableB_window_size_analysis.csv` and `tableE_epoch_analysis.csv` respectively.
Graphing is done in `window_size_analysis_graphing.qmd`.
=#

using CSV, DataFrames, HTTP, HypothesisTests, Base.Threads, Statistics, DataStructures

println(Threads.nthreads())

# Types
@kwdef struct CorTestResult
    Parameter1::Symbol = Symbol()
    Parameter2::Symbol = Symbol()
    r::Float64 = 0.0
    time::Float64 = 0.0
    condition::Symbol = Symbol()
    window_size::Float64 = 0.0
    p::Float64 = 0.0
    t::Float64 = 0.0
    ci_lower::Float64 = 0.0
    ci_higher::Float64 = 0.0
    ci::Float64 = 0.95
    n::Int = 0
    df_error::Int = 0
end

@kwdef struct EpochCorTestResult
    Parameter1::Symbol = Symbol()
    Parameter2::Symbol = Symbol()
    r::Float64 = 0.0
    p::Float64 = 0.0
    t::Float64 = 0.0
    ci_lower::Float64 = 0.0
    ci_higher::Float64 = 0.0
    ci::Float64 = 0.95
    n::Int = 0
    df_error::Int = 0
    epoch::Int = 0
    condition::Symbol = Symbol()
    name::Symbol = Symbol()
end

struct EpochStats
    min_count::Int
    max_count::Int
    mean_count::Float64
    std_count::Float64
end

struct AgeStats
    min::Int
    max::Int
    mean::Float64
    std::Float64
    vec::Vector{Union{Int, Missing}}
end

struct PPStats
    included::Vector{String7}
    included_count::Int
    excluded::Vector{String7}
    excluded_count::Int
    total::Int
    age::AgeStats
    gender::Dict

    function PPStats(included::Vector{String7}, excluded::Vector{String7}, age::AgeStats, gender::Dict)
        unique_included = unique(included)
        unique_excluded = unique(excluded)
        total = length(unique_included) + length(unique_excluded)
        new(unique_included, length(unique_included), unique_excluded, length(unique_excluded), total, age, gender)
    end
end

# Functions

"""
Eagerly evaluate the iterator from `skipmissings` in order to satisfy type checking.
"""
filter_missings(args...) = collect.(skipmissings(args...))

remove_undef(vec) = [vec[i] for i = 1:length(vec) if isassigned(vec, i)] # Remove any undef

function load_csvs(first::Union{Int, Nothing} = nothing)
    csvs = filter(file -> contains(file, "data_hep"), readdir("data", join = true))
    hepdata = reduce(
        vcat,
        [CSV.read(filepath, DataFrame, types = Dict(:Condition => Symbol), missingstring = "NA") for filepath in csvs],
    )

    ppreq = HTTP.get(
        "https://raw.githubusercontent.com/RealityBending/InteroceptionPrimals/main/data/data_participants.csv",
    )
    interoprimals = DataFrame(CSV.File(ppreq.body, missingstring = "NA"), copycols = false)
    rename!(interoprimals, :participant_id => :Participant)

    # Copy a slice for pp stats for later
    pp_population_stats = interoprimals[:, [:Participant, :Age, :Gender]]

    # We only care about MAIA, IAS, HRV and HCT, remove the rest.
    filtered_cols = filter(
        col ->
            !contains(col, r"\d") && (
                col == "Participant" ||
                startswith(col, "MAIA_") ||
                startswith(col, "IAS") ||
                startswith(col, "HRV") ||
                startswith(col, "HCT")
            ),
        names(interoprimals),
    )

    filtered_intero = (@view interoprimals[!, filtered_cols])

    joined = innerjoin(hepdata, filtered_intero, on = :Participant)
    nomissing = dropmissing(joined)

    summary_stats(filtered_intero, nomissing, pp_population_stats, hepdata)

    if (first isa Int)
        return first(nomissing, first)
    else
        return nomissing
    end
end

# Calculate summary statistics. We don't take hepdata because it doesn't have any missing values
function summary_stats(filtered_intero, nomissing, pp_population_stats, hepdata)
    # Create a dataframe showing participants with any columns containing missing data. These pps are excluded.
    global missingdf = filter(row -> any(ismissing, row) && row.Participant in hepdata.Participant, filtered_intero)
    cols_to_remove = [col for col in names(missingdf) if all(!ismissing, missingdf[!, col])] # Remove columns with no missing data, for visualisation purposes
    select!(missingdf, :Participant, Not(cols_to_remove))

    included_participants = unique(nomissing.Participant)
    excluded_participants = unique(missingdf.Participant)

    filter!(row -> row.Participant in included_participants, pp_population_stats)
    age_vec = skipmissing(pp_population_stats.Age)

    agestats = AgeStats(minimum(age_vec), maximum(age_vec), mean(age_vec), std(age_vec), pp_population_stats.Age)
    genderstats = Dict(counter(pp_population_stats.Gender))
    global ppstats = PPStats(included_participants, excluded_participants, agestats, genderstats)

    println("Beginning analysis with $(nrow(nomissing)) time points from $(ppstats.included_count) participants.")
    println("Excluded $(ppstats.excluded_count) participants: $(join(ppstats.excluded, ", "))")
end

function epoch_stats(df)

    grouped = groupby(df, :Participant)
    epoch_summary = combine(grouped, :epoch => maximum => :epoch_count)

    min_count = minimum(epoch_summary.epoch_count)
    max_count = maximum(epoch_summary.epoch_count)
    mean_count = mean(epoch_summary.epoch_count)
    std_count = std(epoch_summary.epoch_count)

    return EpochStats(min_count, max_count, mean_count, std_count)
end

function prepare_variables(windowed)

    time_grouped = groupby(windowed, [:time, :Participant])

    time_combined = combine(
        time_grouped,
        [:AF7, :AF8] .=> mean .=> [:AF7_Mean, :AF8_Mean],
        #    [:AF7, :AF8] .=> median .=> [:AF7_Median, :AF8_Median],
        Cols(startswith("MAIA")) .=> first,
        Cols(startswith("IAS")) .=> first,
        Cols(startswith("HRV")) .=> first,
        Cols(startswith("HCT")) .=> first,
        renamecols = false,
    )

    participant_grouped = groupby(time_combined, :Participant)

    participant_combined = combine(
        participant_grouped,
        [:AF7_Mean, :AF8_Mean] .=> mean,
        #[:AF7_Median, :AF8_Median] .=> median,
        Cols(startswith("MAIA")) .=> first,
        Cols(startswith("IAS")) .=> first,
        Cols(startswith("HRV")) .=> first,
        Cols(startswith("HCT")) .=> first,
        renamecols = false,
    )

    means_medians = eachcol(select(participant_combined, [
        :AF7_Mean,
        :AF8_Mean,
        # :AF7_Median, :AF8_Median
    ], copycols = false))

    interoceptive = eachcol(
        select(
            participant_combined,
            Cols(startswith("MAIA"), startswith("IAS"), startswith("HRV"), startswith("HCT")),
            copycols = false,
        ),
    )

    return (means_medians, interoceptive)
end

function pairwise_correlation(
    means_medians,
    interoceptive,
    output,
    prealloc_i;
    i::Union{Float64, Nothing} = nothing,
    c::Union{Symbol, Nothing} = nothing,
    w::Union{Float64, Nothing} = nothing,
    e::Union{Int, Nothing} = nothing,
    name::Union{Symbol, Nothing} = nothing,
)

    for (Parameter1, cols1) in pairs(means_medians)
        for (Parameter2, cols2) in pairs(interoceptive) # Correlate the AF7/8 means against each interoceptive measure

            (cols1_filtered, cols2_filtered) = filter_missings(cols1, cols2)

            if ((length(cols1_filtered)) === 0 || (length(cols1_filtered)) === 0)
                continue
            end

            cor = HypothesisTests.CorrelationTest(cols1_filtered, cols2_filtered)

            if (cor.r ≈ 1.0 || cor.r ≈ -1.0 || isnan(cor.r) || cor.t ≈ Inf || cor.t ≈ -Inf)
                # Assume something's gone wrong
                continue
            end

            pvalue = HypothesisTests.pvalue(cor)
            (ci_lower, ci_higher) = HypothesisTests.confint(cor)

            if (e === nothing) # If sliding window

                output[prealloc_i[]] = CorTestResult(
                    Parameter1 = Parameter1,
                    Parameter2 = Parameter2,
                    r = cor.r,
                    time = i,
                    condition = c,
                    window_size = w,
                    p = pvalue,
                    t = cor.t,
                    ci_lower = ci_lower,
                    ci_higher = ci_higher,
                    ci = 0.95,
                    n = cor.n,
                    df_error = cor.n - 2, # n minus number of variables
                )

            else # If epoch analysis
                output[prealloc_i[]] = EpochCorTestResult(
                    Parameter1 = Parameter1,
                    Parameter2 = Parameter2,
                    r = cor.r,
                    p = pvalue,
                    t = cor.t,
                    ci_lower = ci_lower,
                    ci_higher = ci_higher,
                    ci = 0.95,
                    n = cor.n,
                    df_error = cor.n - 2, # n minus number of variables
                    epoch = e,
                    condition = c,
                    name = name,
                )
            end

            atomic_add!(prealloc_i, 1)

        end
    end
end

function sliding_window_analysis(df, conditions)
    # 0.05 means 0.05 each side of the time point (total width 0.1)
    # Times are fractions of seconds, so 0.05 = 50ms
    window_widths = range(start = 0.05, stop = 0.2, step = 0.025)
    times = range(start = -0.4, stop = 0.8, step = 0.01)

    steps = length(window_widths) * length(conditions) * length(times) * 52 # 50 correlations for each combination, plus 2 for good luck

    output = Vector{CorTestResult}(undef, steps)
    prealloc_i = Threads.Atomic{Int}(1)

    Threads.@threads for w in window_widths
        for c in conditions
            for i in times

                if (i - w < -0.4 || i + w > 0.8)
                    continue
                end

                windowed = (@view df[(df.time .>= (i - w) .&& df.time .< (i + w)) .&& df.Condition .=== c, :])

                if (nrow(windowed) === 0)
                    # println("Thread $(Threads.threadid()): SKIPPED $i, $c, $w")
                    continue
                end

                (means_medians, interoceptive) = prepare_variables(windowed)

                pairwise_correlation(means_medians, interoceptive, output, prealloc_i; i = i, c = c, w = w)

                println("Thread $(Threads.threadid()): $i, $w, $c")
            end
        end
    end

    global window_df = DataFrame(remove_undef(output))
end

function epoch_analysis(df, conditions)
    global epstats = epoch_stats(df)

    epochs = 1:(epstats.max_count)

    before = df[(df.time .>= -0.225 .&& df.time .< -0.025), :]
    after_early = df[(df.time .>= 0.025 .&& df.time .< 0.225), :]
    after_late = df[(df.time .>= 0.410 .&& df.time .< 0.620), :]

    dfs = (before = before, after_early = after_early, after_late = after_late)

    steps = length(epochs) * length(conditions) * length(dfs) * 52  # 50 correlations for each combination, plus 2 for good luck

    output = Vector{EpochCorTestResult}(undef, steps)
    prealloc_i = Threads.Atomic{Int}(1)

    Threads.@threads for e in epochs
        for (name, df) in pairs(dfs)
            for c in conditions

                windowed = (@view df[df.epoch .<= e .&& df.Condition .=== c, :])

                if (nrow(windowed) === 0)
                    continue
                end

                (means_medians, interoceptive) = prepare_variables(windowed)

                pairwise_correlation(means_medians, interoceptive, output, prealloc_i; e = e, c = c, name = name)

                println("Thread $(Threads.threadid()): $e, $c, $name")
            end
        end
    end
    global epoch_df = DataFrame(remove_undef(output))
end

# Count how many significant correlations there are per mean/index
function significance_analysis()
    significant = (@view window_df[window_df.p .< 0.05, :])

    global significance_count_total = combine(groupby(significant, [:Parameter1, :Parameter2]), nrow => :count)
    global window_significance_count =
        combine(groupby(significant, [:Parameter1, :Parameter2, :window_size]), nrow => :count)
    global condition_significance_count =
        combine(groupby(significant, [:Parameter1, :Parameter2, :condition]), nrow => :count)
    global total_condition_significance = combine(groupby(significant, [:condition]), nrow => :count)

    by_window = groupby(window_df, :window_size)

    abs_mean(x) = mean(abs.(x))
    global abs_mean_correlations = combine(by_window, :r => abs_mean, :r => std)
end

# Main

function main(; do_sliding_window, do_epoch_analysis)

    df = load_csvs()
    println("CSVs loaded.")

    conditions = [:HCT, :RestingState]

    if (do_sliding_window)
        sliding_window_analysis(df, conditions)
        CSV.write("correlations_julia.csv", window_df)
        significance_analysis()
    end

    if (do_epoch_analysis)
        epoch_analysis(df, conditions)
        CSV.write("epoch_correlations_julia.csv", epoch_df)
    end

end

@time main(do_sliding_window = true, do_epoch_analysis = true)