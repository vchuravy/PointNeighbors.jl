using Plots
using BenchmarkTools
import CellListMap
using Morton

# Generate a rectangular point cloud
include("../test/point_cloud.jl")

function point_cloud_z_index(n_points_per_dimension;
                     seed = 1, perturbation_factor_position = 1.0)
    # Fixed seed to ensure reproducibility
    Random.seed!(seed)

    n_dims = length(n_points_per_dimension)
    coordinates = Array{Float64}(undef, n_dims, prod(n_points_per_dimension))
    cartesian_indices = CartesianIndices(n_points_per_dimension)

    for i in axes(coordinates, 2)
        coordinates[:, i] .= Tuple(cartesian_indices[i])
    end

    min_corner = minimum(coordinates, dims = 2)

    function z_index(coords)
        cartesian3morton(SVector(Tuple(PointNeighbors.floor_to_int.((coords .- min_corner) ./ (3, 3, 3))) .+ 1))
    end

    coords_svector = reinterpret(SVector{3, Float64}, coordinates)
    perm = coords_svector .|> z_index |> vec |> sortperm
    coordinates2 = coordinates[:, perm]

    perturb!(coordinates2, perturbation_factor_position * 0.5)

    return coordinates2
end

"""
    plot_benchmarks(benchmark, n_points_per_dimension, iterations;
                    seed = 1, perturbation_factor_position = 1.0,
                    parallel = true, title = "")

Run a benchmark with several neighborhood searches multiple times for increasing numbers
of points and plot the results.

# Arguments
- `benchmark`:              The benchmark function. See [`benchmark_count_neighbors`](@ref)
                            and [`benchmark_n_body`](@ref).
- `n_points_per_dimension`: Initial resolution as tuple. The product is the initial number
                            of points. For example, use `(100, 100)` for a 2D benchmark or
                            `(10, 10, 10)` for a 3D benchmark.
- `iterations`:             Number of refinement iterations

# Keywords
- `parallel = true`:        Loop over all points in parallel
- `title = ""`:             Title of the plot
- `seed = 1`:               Seed to perturb the point positions. Different seeds yield
                            slightly different point positions.
- `perturbation_factor_position = 1.0`: Perturb point positions by this factor. A factor of
                                        `1.0` corresponds to points being moved by
                                        a maximum distance of `0.5` along each axis.

# Examples
```julia
include("benchmarks/benchmarks.jl")

plot_benchmarks(benchmark_count_neighbors, (10, 10), 3)
"""
function plot_benchmarks(benchmark, n_points_per_dimension, iterations;
                         parallel = true, title = "",
                         seed = 1, perturbation_factor_position = 1.0)
    neighborhood_searches_names = [#"GridNeighborhoodSearch";;
                                   #"GridNeighborhoodSearch with `PointWithCoordinates`";;
                                #    "GNHS with `FullGridCellList`";;
                                #    "GNHS with `FullGridCellList` and `PointWithCoordinates`";;]
                                #   "PrecomputedNeighborhoodSearch";;
                                #   "CellListMap.jl";;]
                                    "original";;]
                                    # "z-index";;]

    # Multiply number of points in each iteration (roughly) by this factor
    scaling_factor = 4
    per_dimension_factor = scaling_factor^(1 / length(n_points_per_dimension))
    sizes = [round.(Int, n_points_per_dimension .* per_dimension_factor^(iter - 1))
             for iter in 1:iterations]

    n_particles_vec = prod.(sizes)
    times = zeros(iterations, length(neighborhood_searches_names))

    for iter in 1:iterations
        coordinates = point_cloud(sizes[iter], seed = seed,
                                  perturbation_factor_position = perturbation_factor_position)
        # coords_z_index = point_cloud_z_index(sizes[iter], seed = seed,
        #                           perturbation_factor_position = perturbation_factor_position)

        search_radius = 3.0
        NDIMS = size(coordinates, 1)
        n_particles = size(coordinates, 2)

        min_corner = minimum(coordinates, dims = 2) .- 1.02search_radius
        max_corner = maximum(coordinates, dims = 2) .+ 1.02search_radius

        neighborhood_searches = [
            # TrivialNeighborhoodSearch{NDIMS}(; search_radius, eachpoint = 1:n_particles),
            # GridNeighborhoodSearch{NDIMS}(; search_radius, n_points = n_particles, update_strategy=nothing),
            # GridNeighborhoodSearch{NDIMS}(; cell_list = DictionaryCellList{PointNeighbors.PointWithCoordinates{NDIMS, Float64}, NDIMS}(),
            #                               search_radius, n_points = n_particles, update_strategy=nothing),
            GridNeighborhoodSearch{NDIMS}(; cell_list = FullGridCellList(; search_radius,
                                                                         min_corner,
                                                                         max_corner),
                                          search_radius, n_points = n_particles, update_strategy=nothing),
            # GridNeighborhoodSearch{NDIMS}(; cell_list = FullGridCellList(; search_radius,
            #                                                              min_corner,
            #                                                              max_corner),
            #                               search_radius, n_points = n_particles, update_strategy=nothing),
            # GridNeighborhoodSearch{NDIMS}(; cell_list = FullGridCellList(; search_radius,
            #                                                              min_corner,
            #                                                              max_corner, backend=PointNeighbors.DynamicVectorOfVectors{PointNeighbors.PointWithCoordinates{3, Float64}}),
            #                               search_radius, n_points = n_particles, update_strategy=nothing),
            # PrecomputedNeighborhoodSearch{NDIMS}(; search_radius, n_points = n_particles, update_strategy=nothing),
            # CellListMapNeighborhoodSearch(NDIMS, search_radius),
        ]

        for i in eachindex(neighborhood_searches)
            # if i == 2
            #     coordinates = coords_z_index
            # end
            neighborhood_search = neighborhood_searches[i]
            initialize!(neighborhood_search, coordinates, coordinates)

            time = benchmark(neighborhood_search, coordinates, parallel = parallel)
            times[iter, i] = time
            time_string = BenchmarkTools.prettytime(time * 1e9)
            println("$(neighborhood_searches_names[i])")
            println("with $(join(sizes[iter], "x")) = $(prod(sizes[iter])) particles finished in $time_string\n")
        end
    end

    return n_particles_vec, times, neighborhood_searches_names
    # plot(n_particles_vec, times,
    #      xaxis = :log, yaxis = :log,
    #      xticks = (n_particles_vec, n_particles_vec),
    #      xlabel = "#particles", ylabel = "Runtime [s]",
    #      legend = :outerright, size = (750, 400), dpi = 200,
    #      label = neighborhood_searches_names, title = title)
end
