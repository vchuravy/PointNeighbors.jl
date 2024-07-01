var documenterSearchIndex = {"docs":
[{"location":"license/","page":"License","title":"License","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/LICENSE.md\"","category":"page"},{"location":"license/#License","page":"License","title":"License","text":"","category":"section"},{"location":"license/","page":"License","title":"License","text":"MIT LicenseCopyright (c) 2023-present The TrixiParticles.jl Authors (see Authors) \nCopyright (c) 2023-present Helmholtz-Zentrum hereon GmbH, Institute of Surface Science \n \nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.","category":"page"},{"location":"authors/","page":"Authors","title":"Authors","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/AUTHORS.md\"","category":"page"},{"location":"authors/#Authors","page":"Authors","title":"Authors","text":"","category":"section"},{"location":"authors/","page":"Authors","title":"Authors","text":"This package is maintained by the authors of TrixiParticles.jl. For a full list of authors, see AUTHORS.md in the TrixiParticles.jl repository. These authors form \"The TrixiParticles.jl Authors\", as mentioned under License.","category":"page"},{"location":"reference/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"reference/","page":"API reference","title":"API reference","text":"CurrentModule = PointNeighbors","category":"page"},{"location":"reference/","page":"API reference","title":"API reference","text":"Modules = [PointNeighbors]","category":"page"},{"location":"reference/#PointNeighbors.DictionaryCellList","page":"API reference","title":"PointNeighbors.DictionaryCellList","text":"DictionaryCellList{NDIMS}()\n\nA simple cell list implementation where a cell index (i, j) or (i, j, k) is mapped to a Vector{Int} by a Dict. By using a dictionary, which only stores non-empty cells, the domain is potentially infinite.\n\nThis implementation is very simple, but it neither uses an optimized hash function for integer tuples, nor does it use a contiguous memory layout. Consequently, this cell list is not GPU-compatible.\n\nArguments\n\nNDIMS: Number of dimensions.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.FullGridCellList","page":"API reference","title":"PointNeighbors.FullGridCellList","text":"FullGridCellList(; min_corner, max_corner, search_radius = 0.0,\n                 periodicity = false, backend = DynamicVectorOfVectors{Int32},\n                 max_points_per_cell = 100)\n\nA simple cell list implementation where each (empty or non-empty) cell of a rectangular (axis-aligned) domain is assigned a list of points. This cell list only works when all points are inside the specified domain at all times.\n\nOnly set min_corner and max_corner and use the default values for the other arguments to create an empty \"template\" cell list that can be used to create an empty \"template\" neighborhood search. See copy_neighborhood_search for more details.\n\nKeywords\n\nmin_corner: Coordinates of the domain corner in negative coordinate directions.\nmax_corner: Coordinates of the domain corner in positive coordinate directions.\nsearch_radius = 0.0: Search radius of the neighborhood search, which will determine the                        cell size. Use the default of 0.0 to create a template (see above).\nperiodicity = false: Set to true when using a PeriodicBox with the                        neighborhood search. When using copy_neighborhood_search,                        this option can be ignored an will be set automatically depending                        on the periodicity of the neighborhood search.\nbackend = DynamicVectorOfVectors{Int32}: Type of the data structure to store the actual   cell lists. Can be\nVector{Vector{Int32}}: Scattered memory, but very memory-efficient.\nDynamicVectorOfVectors{Int32}: Contiguous memory, optimizing cache-hits.\nmax_points_per_cell = 100: Maximum number of points per cell. This will be used to                              allocate the DynamicVectorOfVectors. It is not used with                              the Vector{Vector{Int32}} backend.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.GridNeighborhoodSearch","page":"API reference","title":"PointNeighbors.GridNeighborhoodSearch","text":"GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,\n                              periodic_box = nothing,\n                              cell_list = DictionaryCellList{NDIMS}(),\n                              update_strategy = nothing)\n\nSimple grid-based neighborhood search with uniform search radius. The domain is divided into a regular grid. For each (non-empty) grid cell, a list of points in this cell is stored. Instead of representing a finite domain by an array of cells, a potentially infinite domain is represented by storing cell lists in a hash table (using Julia's Dict data structure), indexed by the cell index tuple\n\nleft( leftlfloor fracxd rightrfloor leftlfloor fracyd rightrfloor right) quad textor quad\nleft( leftlfloor fracxd rightrfloor leftlfloor fracyd rightrfloor leftlfloor fraczd rightrfloor right)\n\nwhere x y z are the space coordinates and d is the search radius.\n\nTo find points within the search radius around a position, only points in the neighboring cells are considered.\n\nSee also (Chalela et al., 2021), (Ihmsen et al. 2011, Section 4.4).\n\nAs opposed to (Ihmsen et al. 2011), we do not sort the points in any way, since not sorting makes our implementation a lot faster (although less parallelizable).\n\nArguments\n\nNDIMS: Number of dimensions.\n\nKeywords\n\nsearch_radius = 0.0:    The fixed search radius. The default of 0.0 is useful together                           with copy_neighborhood_search.\nn_points = 0:           Total number of points. The default of 0 is useful together                           with copy_neighborhood_search.\nperiodic_box = nothing: In order to use a (rectangular) periodic domain, pass a                           PeriodicBox.\ncell_list:              The cell list that maps a cell index to a list of points inside                           the cell. By default, a DictionaryCellList is used.\nupdate_strategy = nothing: Strategy to parallelize update!. Available options are:\nnothing: Automatically choose the best available option.\nParallelUpdate(): This is not available for all cell list implementations,   but is the default when available.\nSemiParallelUpdate(): This is available for all cell list implementations   and is the default when ParallelUpdate is not available.\nSerialUpdate()\n\nReferences\n\nM. Chalela, E. Sillero, L. Pereyra, M.A. Garcia, J.B. Cabral, M. Lares, M. Merchán. \"GriSPy: A Python package for fixed-radius nearest neighbors search\". In: Astronomy and Computing 34 (2021). doi: 10.1016/j.ascom.2020.100443\nMarkus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner. \"A Parallel SPH Implementation on Multi-Core CPUs\". In: Computer Graphics Forum 30.1 (2011), pages 99–112. doi: 10.1111/J.1467-8659.2010.01832.X\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.ParallelUpdate","page":"API reference","title":"PointNeighbors.ParallelUpdate","text":"ParallelUpdate()\n\nFully parallel update by using atomic operations to avoid race conditions when adding points into the same cell. This is not available for all cell list implementations, but is the default when available.\n\nSee GridNeighborhoodSearch for usage information.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.PeriodicBox","page":"API reference","title":"PointNeighbors.PeriodicBox","text":"PeriodicBox(; min_corner, max_corner)\n\nDefine a rectangular (axis-aligned) periodic domain.\n\nKeywords\n\nmin_corner: Coordinates of the domain corner in negative coordinate directions.\nmax_corner: Coordinates of the domain corner in positive coordinate directions.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.PrecomputedNeighborhoodSearch","page":"API reference","title":"PointNeighbors.PrecomputedNeighborhoodSearch","text":"PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,\n                                     periodic_box = nothing, update_strategy = nothing)\n\nNeighborhood search with precomputed neighbor lists. A list of all neighbors is computed for each point during initialization and update. This neighborhood search maximizes the performance of neighbor loops at the cost of a much slower update!.\n\nA GridNeighborhoodSearch is used internally to compute the neighbor lists during initialization and update.\n\nArguments\n\nNDIMS: Number of dimensions.\n\nKeywords\n\nsearch_radius = 0.0:    The fixed search radius. The default of 0.0 is useful together                           with copy_neighborhood_search.\nn_points = 0:           Total number of points. The default of 0 is useful together                           with copy_neighborhood_search.\nperiodic_box = nothing: In order to use a (rectangular) periodic domain, pass a                           PeriodicBox.\nupdate_strategy:        Strategy to parallelize update! of the internally used                           GridNeighborhoodSearch. See GridNeighborhoodSearch                           for available options.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.SemiParallelUpdate","page":"API reference","title":"PointNeighbors.SemiParallelUpdate","text":"SemiParallelUpdate()\n\nLoop over all cells in parallel to mark cells with points that now belong to a different cell. Then, move points of affected cells serially to avoid race conditions. This is available for all cell list implementations and is the default when ParallelUpdate is not available.\n\nSee GridNeighborhoodSearch for usage information.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.SerialUpdate","page":"API reference","title":"PointNeighbors.SerialUpdate","text":"SerialUpdate()\n\nDeactivate parallelization in the neighborhood search update. Parallel neighborhood search update can be one of the largest sources of error variations between simulations with different thread numbers due to neighbor ordering changes.\n\nSee GridNeighborhoodSearch for usage information.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.TrivialNeighborhoodSearch","page":"API reference","title":"PointNeighbors.TrivialNeighborhoodSearch","text":"TrivialNeighborhoodSearch{NDIMS}(; search_radius = 0.0, eachpoint = 1:0,\n                                 periodic_box = nothing)\n\nTrivial neighborhood search that simply loops over all points.\n\nArguments\n\nNDIMS: Number of dimensions.\n\nKeywords\n\nsearch_radius = 0.0:    The fixed search radius. The default of 0.0 is useful together                           with copy_neighborhood_search.\neachpoint = 1:0:        Iterator for all point indices. Usually just 1:n_points.                           The default of 1:0 is useful together with                           copy_neighborhood_search.\nperiodic_box = nothing: In order to use a (rectangular) periodic domain, pass a                           PeriodicBox.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.copy_neighborhood_search-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.copy_neighborhood_search","text":"copy_neighborhood_search(search::AbstractNeighborhoodSearch, search_radius, n_points;\n                         eachpoint = 1:n_points)\n\nCreate a new uninitialized neighborhood search of the same type and with the same configuration options as search, but with a different search radius and number of points.\n\nThe TrivialNeighborhoodSearch also requires an iterator eachpoint, which most of the time will be 1:n_points. If the TrivialNeighborhoodSearch is never going to be used, the keyword argument eachpoint can be ignored.\n\nThis is useful when a simulation code requires multiple neighborhood searches of the same kind. One can then just pass an empty neighborhood search as a template and use this function inside the simulation code to generate similar neighborhood searches with different search radii and different numbers of points.\n\n# Template\nnhs = GridNeighborhoodSearch{2}()\n\n# Inside the simulation code, generate similar neighborhood searches\nnhs1 = copy_neighborhood_search(nhs, 1.0, 100)\n\n# output\nGridNeighborhoodSearch{2, Float64, ...}(...)\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.foreach_point_neighbor-Union{Tuple{T}, Tuple{T, Any, Any, Any}} where T","page":"API reference","title":"PointNeighbors.foreach_point_neighbor","text":"foreach_point_neighbor(f, system_coords, neighbor_coords, neighborhood_search;\n                       points = axes(system_coords, 2), parallel = true)\n\nLoop for each point in system_coords over all points in neighbor_coords whose distances to that point are smaller than the search radius and execute the function f(i, j, x, y, d), where\n\ni is the column index of the point in system_coords,\nj the column index of the neighbor in neighbor_coords,\nx an SVector of the coordinates of the point (system_coords[:, i]),\ny an SVector of the coordinates of the neighbor (neighbor_coords[:, j]),\nd the distance between x and y.\n\nThe neighborhood_search must have been initialized or updated with system_coords as first coordinate array and neighbor_coords as second coordinate array.\n\nNote that system_coords and neighbor_coords can be identical.\n\nArguments\n\nf: The function explained above.\nsystem_coords: A matrix where the i-th column contains the coordinates of point i.\nneighbor_coords: A matrix where the j-th column contains the coordinates of point j.\nneighborhood_search: A neighborhood search initialized or updated with system_coords                        as first coordinate array and neighbor_coords as second                        coordinate array.\n\nKeywords\n\npoints: Loop over these point indices. By default all columns of system_coords.\nparallel=true: Run the outer loop over points thread-parallel.\n\nSee also initialize!, update!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.initialize!-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.initialize!","text":"initialize!(search::AbstractNeighborhoodSearch, x, y)\n\nInitialize a neighborhood search with the two coordinate arrays x and y.\n\nIn general, the purpose of a neighborhood search is to find for one point in x all points in y whose distances to that point are smaller than the search radius. x and y are expected to be matrices, where the i-th column contains the coordinates of point i. Note that x and y can be identical.\n\nSee also update!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.update!-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.update!","text":"update!(search::AbstractNeighborhoodSearch, x, y; points_moving = (true, true))\n\nUpdate an already initialized neighborhood search with the two coordinate arrays x and y.\n\nLike initialize!, but reusing the existing data structures of the already initialized neighborhood search. When the points only moved a small distance since the last update! or initialize!, this is significantly faster than initialize!.\n\nNot all implementations support incremental updates. If incremental updates are not possible for an implementation, update! will fall back to a regular initialize!.\n\nSome neighborhood searches might not need to update when only x changed since the last update! or initialize! and y did not change. Pass points_moving = (true, false) in this case to avoid unnecessary updates. The first flag in points_moving indicates if points in x are moving. The second flag indicates if points in y are moving.\n\nSee also initialize!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.@threaded-Tuple{Any, Any}","page":"API reference","title":"PointNeighbors.@threaded","text":"@threaded x for ... end\n\nRun either a threaded CPU loop or launch a kernel on the GPU, depending on the type of x. Semantically the same as Threads.@threads when iterating over a AbstractUnitRange but without guarantee that the underlying implementation uses Threads.@threads or works for more general for loops.\n\nThe first argument must either be a KernelAbstractions.Backend or an array from which the backend can be derived to determine if the loop must be run threaded on the CPU or launched as a kernel on the GPU. Passing KernelAbstractions.CPU() will run the GPU kernel on the CPU.\n\nIn particular, the underlying threading capabilities might be provided by other packages such as Polyester.jl.\n\nwarn: Warn\nThis macro does not necessarily work for general for loops. For example, it does not necessarily support general iterables such as eachline(filename).\n\n\n\n\n\n","category":"macro"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/README.md\"","category":"page"},{"location":"#PointNeighbors.jl","page":"Home","title":"PointNeighbors.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Docs-stable) (Image: Docs-dev) (Image: Slack) (Image: Youtube) (Image: Build Status) (Image: Codecov) (Image: SciML Code Style) (Image: License: MIT)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Work in Progress!","category":"page"}]
}
