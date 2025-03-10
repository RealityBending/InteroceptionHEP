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
end

"""
Give each thread a vector to store its results, then we'll concat them at the end
"""
mutable struct InnerOutputVec{T}
    vec::Vector{T}
    count::Int

    InnerOutputVec{T}(length) where {T} = new(Vector{T}(undef, cld(length, Threads.nthreads())), 1)
end

create_output_vec(length, T) = [InnerOutputVec{T}(length) for _ = 1:Threads.nthreads()]

# Functions

"""
Evaluate the iterator from `skipmissings` ahead of time in order to satisfy type checking.
"""
function filter_missings(args...)
    return collect.(skipmissings(args...))
end

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

    joined = rightjoin(hepdata, interoprimals[!, filtered_cols], on = :Participant)

    if (first isa Int)
        return first(joined, first)
    else
        return joined
    end
end

function prepare_variables(windowed)

    grouped = groupby(windowed, [:Participant])

    # Add the means for each participant to the respective rows, the number of rows remains the same.
    # transformed = transform(grouped, [:AF7, :AF8] .=> mean .=> [:AF7_Mean, :AF8_Mean])

    # Alternatively, create a table with one row per participant, and their associated statistics.
    # Out of these two options, I don't know which is correct. Both fail when doing the epoch analysis within participant.
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
    output;
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

            output.vec[output.count] = CorTestResult(
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
            output.vec[output.count] = EpochCorTestResult(
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
            )
        end

        output.count += 1

    end
end

# Main

function main(; epoch_analysis = false)

    df = load_csvs()
    println("CSVs loaded.")

    conditions = [:HCT, :RestingState]
    times = range(start = -0.4, stop = 0.8, step = 0.01)

    if (epoch_analysis)
        epochs = unique(df.epoch)
        epochs = epochs[epochs .!= 0] # Remove any 0 epochs

        steps = length(epochs) * length(conditions) * length(times) * 51 # 50 correlations for each combination, plus 1 for good luck

        output = create_output_vec(steps, EpochCorTestResult)

        Threads.@threads for e in epochs

            windowed = (@view df[df.epoch .<= e, :])

            if (nrow(windowed) === 0)
                continue
            end

            (means, interoceptive) = prepare_variables(windowed)

            pairwise_correlation(means, interoceptive, output[Threads.threadid()]; e = e)

            println("Thread $(Threads.threadid()): $e")
        end

    else

        window_widths = range(start = 0.1, stop = 0.6, step = 0.1)

        steps = length(window_widths) * length(conditions) * length(times) * 51 # 50 correlations for each combination, plus 1 extra for good luck

        output = create_output_vec(steps, CorTestResult)

        # A w of 0.1 means 0.1 each side of the mean (total width 0.2)
        Threads.@threads for w in window_widths
            for c in conditions, i in times

                windowed = (@view df[(df.time .> (i - w) .&& df.time .< (i + w)) .&& df.Condition .=== c, :])

                if (nrow(windowed) === 0)
                    # println("Thread $(Threads.threadid()): SKIPPED $i, $c, $w")
                    continue
                end

                (means, interoceptive) = prepare_variables(windowed)

                pairwise_correlation(means, interoceptive, output[Threads.threadid()]; i = i, c = c, w = w)

                println("Thread $(Threads.threadid()): $i, $w, ($(i - w), $(i + w)), $c")
            end
        end
    end

    return output
end

@time output = main(epoch_analysis = true)

output2 = reduce(vcat, [inner.vec for inner in output])

output3 = [output2[i] for i = 1:length(output2) if isassigned(output2, i)] # Remove any undef

finaldf = DataFrame(output3)
