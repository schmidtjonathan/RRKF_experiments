import NaNStatistics


function train_test_split_indices(N; split=0.5, seed=0)
	Random.seed!(seed)
    rand_index = Random.randperm(N)

    N_tr =  ceil(Int64, N * split)

    return sort(rand_index[1:N_tr]), sort(rand_index[N_tr+1:end])
end


const DATA_DIR = joinpath(abspath(pwd()), "data")
dataset_index = 0


function at_time(arr, t::Int64)
    unique_times = sort(unique(arr[:, TIME_AXIS]))
    t_mask = arr[:, TIME_AXIS] .== unique_times[t]
    return t_mask
end

function at_time(arr, t::Float64)
    t_mask = arr[:, TIME_AXIS] .== t
    return t_mask
end

function lons_at_time(arr, i::Int64)
    return arr[i][:, LONGITUDE_AXIS]
end

function lats_at_time(arr, i::Int64)
    return arr[i][:, LATITUDE_AXIS]
end

function sort_time(arr)
	time_sorted_inds = sortperm(arr[:, TIME_AXIS])
	unique_times = sort(unique(arr[:, TIME_AXIS]))
	return unique_times, time_sorted_inds
end


function load_london_data()
    raw_data = load(joinpath(DATA_DIR, "raw_data_$(dataset_index).jld2"));

    # PROCESS X

    raw_X = raw_data["all"]["X"];

    mean_X, std_X = mean(raw_X; dims=1), std(raw_X; dims=1)

    X_train = copy(raw_X);
    X_test = copy(raw_X);

    X_train = (X_train .- mean_X) ./ std_X
    X_test = (X_test .- mean_X) ./ std_X

    # @assert all(isapprox.(mean(X_train, dims=1), 0.0, atol=1e-12, rtol=1e-12))
    # @assert all(isapprox.(std(X_train, dims=1), 1.0, atol=1e-12, rtol=1e-12))

    train_times, train_time_idcs = sort_time(X_train)
    test_times, test_time_idcs = sort_time(X_test)
    raw_times, raw_time_idcs = sort_time(raw_X)

    @assert train_times == test_times
    # @assert train_times == raw_times
    @assert train_time_idcs == test_time_idcs

    raw_X = raw_X[raw_time_idcs, :]
    X_train = X_train[train_time_idcs, :]
    X_test = X_test[test_time_idcs, :]

    # PROCESS Y
    raw_Y = vec(raw_data["all"]["Y"]);
    min_Y, max_Y = minimum(filter(!isnan, raw_Y)), maximum(filter(!isnan, raw_Y))

    raw_Y = raw_Y[raw_time_idcs]
    raw_Y_mat = RRKF.vecvec2mat([raw_Y[at_time(raw_X, t)] for t in raw_times])
    mean_Y = NaNStatistics.nanmean(raw_Y_mat)
    std_Y = NaNStatistics.nanstd(raw_Y_mat)
    # For the degenerate case that we have no variance in spatial nodes, we have to prevent
    # division by zero
    # replace!(std_Y, 0.0 => 1.0)

    Y_train_mat = copy(raw_Y_mat)
    Y_test_mat = copy(raw_Y_mat)
    Y_train_mat = (Y_train_mat .- mean_Y) ./ std_Y
    Y_test_mat = (Y_test_mat .- mean_Y) ./ std_Y

    for i_y in axes(Y_train_mat, 1)
        cur_train_idcs, cur_test_idcs = train_test_split_indices(length(Y_train_mat[i_y, :]); split=0.8, seed=i_y)
        Y_train_mat[i_y, cur_test_idcs] .= NaN64
        Y_test_mat[i_y, cur_train_idcs] .= NaN64
    end


    # CONVERT EVERYTHING FROM MATRICES TO VECTORS OF VECTORS
    raw_X = [raw_X[at_time(raw_X, t), :] for t in raw_times]
    X_train = [X_train[at_time(X_train, t), :] for t in train_times]
    X_test = [X_test[at_time(X_test, t), :] for t in test_times]

    grid_lons, grid_lats, grid_lons_raw, grid_lats_raw = begin
        # CHECK THAT LONS AND LATS ARE THE SAME AT EVERY TIME POINT AND BETWEEN TRAIN AND TEST
        for i in 2:length(train_times)
            if lons_at_time(X_train, i) != lons_at_time(X_train, i-1)
                error("lons")
            end
            if lats_at_time(X_train, i) != lats_at_time(X_train, i-1)
                error("lats")
            end
            if lons_at_time(raw_X, i) != lons_at_time(raw_X, i-1)
                error("lons")
            end
            if lats_at_time(raw_X, i) != lats_at_time(raw_X, i-1)
                error("lats")
            end
            if lons_at_time(X_test, i) != lons_at_time(X_test, i-1)
                error("lons")
            end
            if lats_at_time(X_test, i) != lats_at_time(X_test, i-1)
                error("lats")
            end
        @assert length(lats_at_time(X_train, i)) == length(lats_at_time(X_train, i-1)) == length(lons_at_time(X_train, i)) == length(lons_at_time(X_train, i-1))
        # @assert length(lats_at_time(X_train, i)) == length(Y_train[i])
        end

        lons_at_time(X_train, 1), lats_at_time(X_train, 1), lons_at_time(raw_X, 1), lats_at_time(raw_X, 1)
    end


    Y_train = [Y_train_mat[rw, :] for rw in axes(Y_train_mat, 1)]
    Y_test = [Y_test_mat[rw, :] for rw in axes(Y_test_mat, 1)]
    raw_Y = [raw_Y_mat[rw, :] for rw in axes(raw_Y_mat, 1)]

    @assert issorted(train_times)
    @assert issorted(raw_times)
    return raw_times / 3600.0, grid_lons, grid_lats, grid_lons_raw, grid_lats_raw, raw_Y, Y_train, Y_test, mean_Y, std_Y, min_Y, max_Y
end
