using CSV, DataFrames, HTTP, HypothesisTests, Base.Threads, Statistics

println(Threads.nthreads())

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

const window_widths = range(start = 0.1, stop = 0.6, step = 0.1)
const conditions = [:HCT, :RestingState]
const times = range(start = -0.4, stop = 0.8, step = 0.01)

const steps = length(window_widths) * length(conditions) * length(times)
const max_corr_count = steps * 50 # 50 correlations for each combination

# Preallocate the vector with a struct of dummy values
output = Vector{CorTestResult}(undef, max_corr_count)

prealloc_i::Threads.Atomic{Int} = Threads.Atomic{Int}(1) # Needed for multi-threading

"""
Evaluate the iterator from `skipmissings` ahead of time in order to satisfy type checking.
"""
function filter_missings(args...)
    return collect.(skipmissings(args...))
end

function load_csvs()
    csvs = filter(file -> contains(file, "data_hep"), readdir("data", join = true))
    hepdata = reduce(
        vcat,
        [CSV.read(filepath, DataFrame, types = Dict(:Condition => Symbol), missingstring = "NA") for filepath in csvs],
    )

    ppreq = HTTP.get(
        "https://raw.githubusercontent.com/RealityBending/InteroceptionPrimals/main/data/data_participants.csv",
    )
    interoprimals = DataFrame(CSV.File(ppreq.body, missingstring = "NA"))
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

    return rightjoin(hepdata, interoprimals[!, filtered_cols], on = :Participant)
end

function prepare_window(df, i, c, w)

    windowed = df[(df.time .> (i - w) .&& df.time .< (i + w)) .&& df.Condition .== c, :]
    grouped = groupby(windowed, :Participant)
    transform!(grouped, [:AF7, :AF8] .=> mean .=> [:AF7_Mean, :AF8_Mean])

    means = eachcol(parent(grouped)[!, [:AF7_Mean, :AF8_Mean]])
    interoceptive = eachcol(
        select!(parent(grouped), Cols(startswith("MAIA"), startswith("IAS"), startswith("HRV"), startswith("HCT"))),
    )

    return (means, interoceptive)
end

function pairwise_correlation(means, interoceptive, i, c, w)

    for (Parameter1, cols1) in pairs(means), (Parameter2, cols2) in pairs(interoceptive) # Correlate the AF7/8 means against each interoceptive measure

        cor = HypothesisTests.CorrelationTest(filter_missings(cols1, cols2)...)
        pvalue = HypothesisTests.pvalue(cor)
        (ci_lower, ci_higher) = HypothesisTests.confint(cor)

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

        atomic_add!(prealloc_i, 1)

    end
end

function sliding_window_analysis()

    df = load_csvs()
    println("CSVs loaded...")

    # @time df = CSV.read("completedf.csv", DataFrame, types = Dict(:Condition => Symbol, :Participant => Symbol)) # "completedf.csv" is a CSV of `dffeat` from `sliding_window_analysis.qmd`, just to save me having to learn how to do that manipulation in Julia

    # A w of 0.1 means 0.1 each side of the mean (total width 0.2)
    Threads.@threads for w in window_widths
        for c in conditions, i in times

            (means, interoceptive) = prepare_window(df, i, c, w)

            pairwise_correlation(means, interoceptive, i, c, w)

            println("Thread $(Threads.threadid()): $i, $c, $w")
        end
    end
end

@time sliding_window_analysis()

output = [output[i] for i = 1:length(output) if isassigned(output, i)] # Remove any undef

finaldf = DataFrame(output)
