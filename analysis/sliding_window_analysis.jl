using CSV, DataFrames, HTTP, HypothesisTests, Base.Threads, Statistics

println(Threads.nthreads())

# Types
@kwdef struct CorTestResult
    Parameter1::Symbol = Symbol()
    Parameter2::Symbol = Symbol()
    r::Float64 = 0.0
    abs_r::Float64 = 0.0
    time::Float64 = 0.0
    condition::Symbol = Symbol()
    window_size::Float64 = 0.0
    p::Float64 = 0.0
    t::Float64 = 0.0
    ci_lower::Float64 = 0.0
    ci_higher::Float64 = 0.0
    ci::Float64 = 0.0
    n::Int = 0
    df_error::Int = 0
end

@kwdef struct EpochCorTestResult
    Parameter1::Symbol = Symbol()
    Parameter2::Symbol = Symbol()
    r::Float64 = 0.0
    abs_r::Float64 = 0.0
    p::Float64 = 0.0
    t::Float64 = 0.0
    ci_lower::Float64 = 0.0
    ci_higher::Float64 = 0.0
    ci::Float64 = 0.0
    n::Int = 0
    df_error::Int = 0
    epoch::Int = 0
    condition::Symbol = Symbol()
end

# Functions

"""
Evaluate the iterator from `skipmissings` ahead of time in order to satisfy type checking.
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

    # Create a dataframe showing participants with any missing data. These pps are excluded.
    global missings = filter(x -> any(ismissing, x), filtered_intero)
    cols_to_remove = [col for col in names(missings) if all(!ismissing, missings[!, col])]
    select!(missings, :Participant, Not(cols_to_remove))

    joined = innerjoin(hepdata, filtered_intero, on = :Participant)
    dropped = dropmissing(joined)

    removed_missing = nrow(joined) - nrow(dropped)
    println("$removed_missing time points with missing values omitted")

    if (first isa Int)
        return first(dropped, first)
    else
        return dropped
    end
end

function prepare_variables(windowed)

    # What do I groupby? Only participant has good r but bad p, Participant and epoch has bad r but good p!
    grouped = groupby(windowed, [:Participant, :epoch])

    combined = combine(
        grouped,
        [:AF7, :AF8] .=> mean .=> [:AF7_Mean, :AF8_Mean],
        Cols(startswith("MAIA")) .=> first,
        Cols(startswith("IAS")) .=> first,
        Cols(startswith("HRV")) .=> first,
        Cols(startswith("HCT")) .=> first,
        renamecols = false,
    )

    means = eachcol(select(combined, [:AF7_Mean, :AF8_Mean], copycols = false))

    interoceptive = eachcol(
        select(
            combined,
            Cols(startswith("MAIA"), startswith("IAS"), startswith("HRV"), startswith("HCT")),
            copycols = false,
        ),
    )

    return (means, interoceptive)
end

function pairwise_correlation(
    means,
    interoceptive,
    output,
    prealloc_i;
    i::Union{Float64, Nothing} = nothing,
    c::Union{Symbol, Nothing} = nothing,
    w::Union{Float64, Nothing} = nothing,
    e::Union{Int, Nothing} = nothing,
)

    for (Parameter1, cols1) in pairs(means), (Parameter2, cols2) in pairs(interoceptive) # Correlate the AF7/8 means against each interoceptive measure

        (cols1_filtered, cols2_filtered) = filter_missings(cols1, cols2)

        if ((length(cols1_filtered)) === 0 || (length(cols1_filtered)) === 0)
            continue
        end

        cor = HypothesisTests.CorrelationTest(cols1_filtered, cols2_filtered)

        if (cor.r ≈ 1.0 || cor.r ≈ -1.0 || isnan(cor.r) || cor.t ≈ Inf)
            # All the epoch correlations get caught here :(
            continue
        end

        pvalue = HypothesisTests.pvalue(cor)
        (ci_lower, ci_higher) = HypothesisTests.confint(cor)

        if (e === nothing) # If sliding window

            output[prealloc_i[]] = CorTestResult(
                Parameter1 = Parameter1,
                Parameter2 = Parameter2,
                r = cor.r,
                abs_r = abs(cor.r),
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
                abs_r = abs(cor.r),
                p = pvalue,
                t = cor.t,
                ci_lower = ci_lower,
                ci_higher = ci_higher,
                ci = 0.95,
                n = cor.n,
                df_error = cor.n - 2, # n minus number of variables
                epoch = e,
                condition = c,
            )
        end

        atomic_add!(prealloc_i, 1)

    end
end

function sliding_window_analysis(df, conditions)
    times = range(start = -0.4, stop = 0.8, step = 0.01)
    window_widths = range(start = 0.05, stop = 0.5, step = 0.05)

    steps = length(window_widths) * length(conditions) * length(times) * 51 # 50 correlations for each combination, plus 1 extra for good luck

    output = Vector{CorTestResult}(undef, steps)
    prealloc_i = Threads.Atomic{Int}(1)

    # A w of 0.1 means 0.1 each side of the mean (total width 0.2)
    Threads.@threads for w in window_widths
        for c in conditions, i in times

            if (i - w < -0.4)
                continue
            end

            windowed = (@view df[(df.time .>= (i - w) .&& df.time .< (i + w)) .&& df.Condition .=== c, :])

            if (nrow(windowed) === 0)
                # println("Thread $(Threads.threadid()): SKIPPED $i, $c, $w")
                continue
            end

            (means, interoceptive) = prepare_variables(windowed)

            pairwise_correlation(means, interoceptive, output, prealloc_i; i = i, c = c, w = w)

            println("Thread $(Threads.threadid()): $i, $w, ($(i - w), $(i + w)), $c")
        end
    end

    global window_df = DataFrame(remove_undef(output))
end

function epoch_analysis(df, conditions)
    epochs = unique(df.epoch)

    steps = length(epochs) * length(conditions) * 51 # 50 correlations for each combination, plus 1 for good luck

    output = Vector{EpochCorTestResult}(undef, steps)
    prealloc_i = Threads.Atomic{Int}(1)

    Threads.@threads for e in epochs
        for c in conditions

            windowed = (@view df[df.epoch .<= e .&& df.Condition .=== c, :])

            if (nrow(windowed) === 0)
                continue
            end

            (means, interoceptive) = prepare_variables(windowed)

            pairwise_correlation(means, interoceptive, output, prealloc_i; e = e, c = c)

            println("Thread $(Threads.threadid()): $e, $c")
        end
    end

    global epoch_df = DataFrame(remove_undef(output))
end
# Main

function main(; do_sliding_window = true, do_epoch_analysis = false)

    df = load_csvs()
    println("CSVs loaded.")

    conditions = [:HCT, :RestingState]

    if (do_sliding_window)
        sliding_window_analysis(df, conditions)
    end

    if (do_epoch_analysis)
        epoch_analysis(df, conditions)
    end

end

@time main(do_sliding_window = true, do_epoch_analysis = true)
