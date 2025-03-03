using CSV, DataFrames, HTTP, HypothesisTests, Base.Threads, Statistics

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
    ci::Float64 = 0.0
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
    ci::Float64 = 0.0
    n::Int = 0
    df_error::Int = 0
    epoch::Int = 0
    participant::String = ""
end

# Functions

"""
Evaluate the iterator from `skipmissings` ahead of time in order to satisfy type checking.
"""
function filter_missings(args...)
    return collect.(skipmissings(args...))
end

function load_csvs(first::Int = nothing)
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
            (
                col == "Participant" ||
                startswith(col, "MAIA_") ||
                startswith(col, "IAS") ||
                startswith(col, "HRV") ||
                startswith(col, "HCT")
            ) && !contains(col, r"\d"),
        names(interoprimals),
    )

    joined = rightjoin(hepdata, interoprimals[!, filtered_cols], on = :Participant)

    if (first isa Int)
        return first(joined, first)
    else
        return joined
    end
end

function prepare_variables(windowed)

    grouped = groupby(windowed, :Participant)

    # Add the means for each participant to the respective rows, the number of rows remains the same.
    # transformed = transform(grouped, [:AF7, :AF8] .=> mean .=> [:AF7_Mean, :AF8_Mean])

    # Alternatively, create a table with one row per participant, and their associated statistics.
    # Out of these two options, I don't know which is correct. Both fail when doing the epoch analysis.
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
    pp::Union{String7, Nothing} = nothing,
    e::Union{Int, Nothing} = nothing,
)

    for (Parameter1, cols1) in pairs(means), (Parameter2, cols2) in pairs(interoceptive) # Correlate the AF7/8 means against each interoceptive measure

        (cols1_filtered, cols2_filtered) = filter_missings(cols1, cols2)

        if ((length(cols1_filtered)) === 0 || (length(cols1_filtered)) === 0)
            continue
        end

        cor = HypothesisTests.CorrelationTest(cols1_filtered, cols2_filtered)

        if (cor.r ≈ 1.0 || cor.r ≈ -1.0 || isnan(cor.r) || cor.t ≈ Inf || cor.t ≈ -1.0 || cor.t ≈ 1.0)
            # All the epoch correlations get caught here :(
            continue
        end

        pvalue = HypothesisTests.pvalue(cor)
        (ci_lower, ci_higher) = HypothesisTests.confint(cor)

        if (pp === nothing && e === nothing)

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

        else
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
                participant = pp,
                epoch = e,
            )
        end

        prealloc_i[] += 1

    end
end

# Main

function main(; epoch_analysis = false)

    df = load_csvs()
    println("CSVs loaded.")

    prealloc_i::Threads.Atomic{Int} = Threads.Atomic{Int}(1) # Needed for multi-threading

    if (epoch_analysis)
        epochs = filter(i -> i != 0, unique(df.epoch)) # Remove 0 because it's meaningless when used in a correlation
        participants = unique(df.Participant)

        steps = length(participants) * length(epochs) * 50 # 50 correlations for each combination

        # Preallocate the vector with a struct of dummy values
        output = Vector{EpochCorTestResult}(undef, steps)

        Threads.@threads for pp in participants
            for e in epochs

                windowed = (@view df[df.Participant .== pp .&& df.epoch .<= e, :])

                if (nrow(windowed) === 0)
                    continue
                end

                (means, interoceptive) = prepare_variables(windowed)

                pairwise_correlation(means, interoceptive, output, prealloc_i; pp = pp, e = e)

                println("Thread $(Threads.threadid()): $pp, $e")

            end
        end

    else

        window_widths = range(start = 0.1, stop = 0.6, step = 0.1)
        conditions = [:HCT, :RestingState]
        times = range(start = -0.4, stop = 0.8, step = 0.01)

        steps = length(window_widths) * length(conditions) * length(times) * 51 # 50 correlations for each combination, plus 1 extra for good luck

        # Preallocate the vector with a struct of dummy values
        output = Vector{CorTestResult}(undef, steps)

        # output = create_output_vec(steps, CorTestResult)

        # A w of 0.1 means 0.1 each side of the mean (total width 0.2)
        Threads.@threads for w in window_widths
            for c in conditions, i in times

                windowed = (@view df[(df.time .> (i - w) .&& df.time .< (i + w)) .&& df.Condition .=== c, :])

                if (nrow(windowed) === 0)
                    # println("Thread $(Threads.threadid()): SKIPPED $i, $c, $w")
                    continue
                end

                (means, interoceptive) = prepare_variables(windowed)

                pairwise_correlation(means, interoceptive, output, prealloc_i; i = i, c = c, w = w)

                println("Thread $(Threads.threadid()): $i, $w, ($(i - w), $(i + w)), $c")
            end
        end
    end

    return output
end

@time output = main(epoch_analysis = false)

output2 = [output[i] for i = 1:length(output) if isassigned(output, i)] # Remove any undef

finaldf = DataFrame(output2)
