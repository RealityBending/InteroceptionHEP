<<<<<<< HEAD
using CSV, DataFrames, HypothesisTests, Base.Threads, Statistics

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
=======
using CSV, DataFrames, TidierData, HypothesisTests, Base.Threads

println(Threads.nthreads())

@time df = CSV.read("completedf.csv", DataFrame) # "completedf.csv" is a CSV of `dffeat` from `sliding_window_analysis.qmd`, just to save me having to learn how to do that manipulation in Julia

window_sizes = range(start=0.1, stop=0.6, step=0.1)
conditions = ["HCT", "RestingState"]
times = range(start=-0.4, stop=0.8, step=0.01)

steps = length(window_sizes) * length(conditions) * length(times)
max_corr_count = steps * 50
output = Vector{Any}(undef, max_corr_count) # 50 correlations for each combination

prealloc_i = Threads.Atomic{Int64}(1) # Needed for multi-threading
>>>>>>> 881669abbec6ac533ca54e5eea792563642a4ab7

"""
Evaluate the iterator from `skipmissings` ahead of time in order to satisfy type checking.
"""
function filter_missings(args...)
    return collect.(skipmissings(args...))
end

<<<<<<< HEAD
function prepare_df(df, i, c, w)

    windowed = df[(df.time .> (i - w) .&& df.time .< (i + w)) .&& df.Condition .== c, :]
    grouped = groupby(windowed, :Participant)
    transform!(grouped, [:AF7, :AF8] .=> mean .=> [:AF7_Mean, :AF8_Mean])

    means = eachcol(parent(grouped)[!, [:AF7_Mean, :AF8_Mean]])
    interoceptive = eachcol(select!(parent(grouped), Cols(startswith.(["MAIA", "IAS", "HRV", "HCT"]))))

    return (means, interoceptive)
end

=======
>>>>>>> 881669abbec6ac533ca54e5eea792563642a4ab7
function pairwise_correlation(means, interoceptive, i, c, w)

    for (Parameter1, cols1) in pairs(means), (Parameter2, cols2) in pairs(interoceptive) # Correlate the AF7/8 means against each interoceptive measure

        cor = HypothesisTests.CorrelationTest(filter_missings(cols1, cols2)...)
        pvalue = HypothesisTests.pvalue(cor)
        (ci_lower, ci_higher) = HypothesisTests.confint(cor)

<<<<<<< HEAD
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

=======
        output[prealloc_i[]] = Dict(
            "Parameter1" => Parameter1,
            "Parameter2" => Parameter2,
            "r" => cor.r,
            "time" => i,
            "condition" => c,
            "window_size" => w,
            "p" => pvalue,
            "t" => cor.t,
            "ci_lower" => ci_lower,
            "ci_higher" => ci_higher,
            "ci" => 0.95,
            "n" => cor.n,
            "df_error" => cor.n - 2 # n minus number of variables
        )
>>>>>>> 881669abbec6ac533ca54e5eea792563642a4ab7
        atomic_add!(prealloc_i, 1)

    end
end

<<<<<<< HEAD
function sliding_window_analysis()
    @time df = CSV.read("completedf.csv", DataFrame, types = Dict(:Condition => Symbol)) # "completedf.csv" is a CSV of `dffeat` from `sliding_window_analysis.qmd`, just to save me having to learn how to do that manipulation in Julia

    # A w of 0.1 means 0.1 each side of the mean (total width 0.2)
    Threads.@threads for w in window_widths
=======
function prepare_df(df, i, c, w)
    local windowed = @chain df begin
        @group_by(Participant) # I don't know why we need to group by Pp, but the correlation returns NA if we don't?
        @filter((time > (!!i - !!w) && time < (!!i + !!w)) && Condition == !!c)
        @mutate(AF7_Mean = mean(AF7), AF8_Mean = mean(AF8))
        @ungroup # Correlation also returns NA if we don't ungroup
    end

    means = eachcol(windowed[:, [:AF7_Mean, :AF8_Mean]])
    interoceptive = eachcol(select(windowed, Cols(startswith("MAIA"), startswith("IAS"), startswith("HRV"), startswith("HCT"))
    ))

    return (means, interoceptive)
end

function sliding_window_analysis()
    # A w of 0.1 means 0.1 each side of the mean (total size 0.2)
    Threads.@threads for w in window_sizes
>>>>>>> 881669abbec6ac533ca54e5eea792563642a4ab7
        for c in conditions, i in times

            (means, interoceptive) = prepare_df(df, i, c, w)

            pairwise_correlation(means, interoceptive, i, c, w)

            println("Thread $(Threads.threadid()): $i, $c, $w")
        end
    end
end

<<<<<<< HEAD
@time sliding_window_analysis()

output = filter(row -> row.n != 0, output) # Remove any uninitialised structs

finaldf = DataFrame(output)
=======

@time sliding_window_analysis()

finaldf = vcat(DataFrame.([output[i] for i in 1:length(output) if isassigned(output, i)])...)
>>>>>>> 881669abbec6ac533ca54e5eea792563642a4ab7
