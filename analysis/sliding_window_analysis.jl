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

"""
Evaluate the iterator from `skipmissings` ahead of time in order to satisfy type checking.
"""
function filter_missings(args...)
    return collect.(skipmissings(args...))
end

function pairwise_correlation(means, interoceptive, i, c, w)

    for (Parameter1, cols1) in pairs(means), (Parameter2, cols2) in pairs(interoceptive) # Correlate the AF7/8 means against each interoceptive measure

        cor = HypothesisTests.CorrelationTest(filter_missings(cols1, cols2)...)
        pvalue = HypothesisTests.pvalue(cor)
        (ci_lower, ci_higher) = HypothesisTests.confint(cor)

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
        atomic_add!(prealloc_i, 1)

    end
end

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
        for c in conditions, i in times

            (means, interoceptive) = prepare_df(df, i, c, w)

            pairwise_correlation(means, interoceptive, i, c, w)

            println("Thread $(Threads.threadid()): $i, $c, $w")
        end
    end
end


@time sliding_window_analysis()

finaldf = vcat(DataFrame.([output[i] for i in 1:length(output) if isassigned(output, i)])...)