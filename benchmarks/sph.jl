using PointNeighbors
using TrixiParticles
using BenchmarkTools

# const TrivialNeighborhoodSearch = PointNeighbors.TrivialNeighborhoodSearch
# const GridNeighborhoodSearch = PointNeighbors.GridNeighborhoodSearch
# const PrecomputedNeighborhoodSearch = PointNeighbors.PrecomputedNeighborhoodSearch

"""
    benchmark_wcsph(neighborhood_search, coordinates; parallel = true)

A benchmark of the right-hand side of a full real-life Weakly Compressible
Smoothed Particle Hydrodynamics (WCSPH) simulation with TrixiParticles.jl.
This method is used to simulate an incompressible fluid.
"""
function benchmark_wcsph(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    v = vcat(fluid.velocity, fluid.density')
    u = copy(coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, neighborhood_search)
    TrixiParticles.compute_pressure!(fluid_system, v)

    return @belapsed $(TrixiParticles.interact!)($dv, $v, $u, $v, $u, $neighborhood_search,
                                              $fluid_system, $fluid_system)
end

function benchmark_wcsph2(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    # v = vcat(fluid.velocity, fluid.density')
    # u = copy(coordinates)
    # dv = zero(v)
    v = Array{Float64, 2}(undef, 4, size(coordinates, 2))
    u = similar(coordinates)
    dv = similar(v)

    PointNeighbors.@threaded v for particle in axes(v, 2)
        for i in axes(u, 1)
            v[i, particle] = fluid.velocity[i, particle]
        end
        v[4, particle] = fluid.density[particle]
    end

    PointNeighbors.@threaded u for particle in axes(u, 2)
        for i in axes(u, 1)
            u[i, particle] = coordinates[i, particle]
        end
    end

    PointNeighbors.@threaded dv for particle in axes(dv, 2)
        for i in axes(dv, 1)
            dv[i, particle] = 0.0
        end
    end

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, neighborhood_search)
    TrixiParticles.compute_pressure!(fluid_system, v)

    return @belapsed $(TrixiParticles.interact!)($dv, $v, $u, $v, $u, $neighborhood_search,
                                              $fluid_system, $fluid_system)
end

function benchmark_wcsph_gpu32(neighborhood_search, coordinates_; parallel = true)
    coordinates = convert(Matrix{Float32}, coordinates_)
    density = 1000f0
    fluid = InitialCondition(; coordinates, density, mass = 0.1f0)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = convert(Float32, PointNeighbors.search_radius(neighborhood_search))
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10f0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02f0, beta = 0.0f0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1f0)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               acceleration = (0f0, 0f0, 0f0),
                                               density_diffusion = density_diffusion)

    system_gpu = PointNeighbors.Adapt.adapt(CuArray, fluid_system)
    nhs2 = copy_neighborhood_search(neighborhood_search, smoothing_length, size(coordinates, 2))
    initialize!(nhs2, coordinates, coordinates)
    nhs_gpu = PointNeighbors.Adapt.adapt(CuArray, nhs2)

    v = cu(vcat(fluid.velocity, fluid.density'))
    u = cu(coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system_gpu, nhs_gpu)
    TrixiParticles.compute_pressure!(system_gpu, v)

    # CUDA.@profile external=true begin
    #     TrixiParticles.interact!(dv, v, u, v, u, nhs_gpu, system_gpu, system_gpu)
    # end

    # @descend TrixiParticles.interact!(dv, v, u, v, u, nhs_gpu, system_gpu, system_gpu)

    # return 1.0
    return @belapsed $(TrixiParticles.interact!)($dv, $v, $u, $v, $u, $nhs_gpu,
                                              $system_gpu, $system_gpu)
end

function benchmark_wcsph_cpu32(neighborhood_search, coordinates_; parallel = true)
    coordinates = convert(Matrix{Float32}, coordinates_)
    density = 1000f0
    fluid = InitialCondition(; coordinates, density, mass = 0.1f0)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = convert(Float32, PointNeighbors.search_radius(neighborhood_search))
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10f0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02f0, beta = 0.0f0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1f0)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               acceleration = (0f0, 0f0, 0f0),
                                               density_diffusion = density_diffusion)

    nhs2 = copy_neighborhood_search(neighborhood_search, smoothing_length, size(coordinates, 2))
    initialize!(nhs2, coordinates, coordinates)

    v = vcat(fluid.velocity, fluid.density')
    u = coordinates
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, nhs2)
    TrixiParticles.compute_pressure!(fluid_system, v)

    # CUDA.@profile external=true begin
    #     TrixiParticles.interact!(dv, v, u, v, u, nhs2, fluid_system, fluid_system)
    # end

    TrixiParticles.interact!(dv, v, u, v, u, nhs2, fluid_system, fluid_system)

    return 1.0
    # return @belapsed $(TrixiParticles.interact!)($dv, $v, $u, $v, $u, $nhs2,
                                            #   $fluid_system, $fluid_system)
end

function benchmark_wcsph_gpu(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    system_gpu = PointNeighbors.Adapt.adapt(CuArray, fluid_system)
    nhs_gpu = PointNeighbors.Adapt.adapt(CuArray, neighborhood_search)

    v = convert(CuArray, vcat(fluid.velocity, fluid.density'))
    u = convert(CuArray, coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system_gpu, nhs_gpu)
    TrixiParticles.compute_pressure!(system_gpu, v)

    # CUDA.@profile external=true begin
    #     TrixiParticles.interact!(dv, v, u, v, u, nhs_gpu, system_gpu, system_gpu)
    # end
    return @belapsed $(TrixiParticles.interact!)($dv, $v, $u, $v, $u, $nhs_gpu,
                                              $system_gpu, $system_gpu)
end

function benchmark_wcsph_amdgpu(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    system_gpu = PointNeighbors.Adapt.adapt(ROCArray, fluid_system)
    nhs_gpu = PointNeighbors.Adapt.adapt(ROCArray, neighborhood_search)

    v = convert(ROCArray, vcat(fluid.velocity, fluid.density'))
    u = convert(ROCArray, coordinates)
    dv = zero(v)

    # Initialize the system
    TrixiParticles.initialize!(system_gpu, nhs_gpu)
    TrixiParticles.compute_pressure!(system_gpu, v)

    TrixiParticles.interact!(dv, v, u, v, u, nhs_gpu, system_gpu, system_gpu)
    return 1.0
    # return @belapsed $(TrixiParticles.interact!)($dv, $v, $u, $v, $u, $nhs_gpu,
    #                                           $system_gpu, $system_gpu)
end

"""
    benchmark_tlsph(neighborhood_search, coordinates; parallel = true)

A benchmark of the right-hand side of a full real-life Total Lagrangian
Smoothed Particle Hydrodynamics (TLSPH) simulation with TrixiParticles.jl.
This method is used to simulate an elastic structure.
"""
function benchmark_tlsph(neighborhood_search, coordinates; parallel = true)
    material = (density = 1000.0, E = 1.4e6, nu = 0.4)
    solid = InitialCondition(; coordinates, density = material.density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    solid_system = TotalLagrangianSPHSystem(solid, smoothing_kernel, smoothing_length,
                                            material.E, material.nu)

    v = copy(solid.velocity)
    u = copy(solid.coordinates)
    dv = zero(v)

    return @belapsed TrixiParticles.interact!($dv, $v, $u, $v, $u, $neighborhood_search,
                                              $solid_system, $solid_system)
end

function benchmark_wcsph_rhs(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    semi = Semidiscretization(fluid_system; neighborhood_search)

    # TrixiParticles.jl creates a copy of the passed neighborhood search
    nhs = TrixiParticles.get_neighborhood_search(fluid_system, semi)
    initialize!(nhs, coordinates, coordinates)

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, nhs)

    v_ode = vec(vcat(fluid.velocity, fluid.density'))
    u_ode1 = vec(coordinates)
    dv_ode = zero(v_ode)
    du_ode = zero(u_ode1)

    # Similar to the alternating update, we simulate a real update in this benchmark
    # by alternating between two slightly different coordinate arrays.

    coordinates2 = copy(coordinates)
    # Perturb all coordinates with a perturbation factor of `0.015`.
    # This factor was tuned so that ~0.5% of the particles change their cell during an
    # update in 2D and ~0.7% in 3D.
    # These values are the same as the experimentally computed averages in 2D and 3D SPH
    # dam break simulations. So this benchmark replicates a real-life SPH update.
    perturb!(coordinates2, 0.015)

    u_ode2 = vec(coordinates2)

    function double_rhs!(dv_ode, du_ode, v_ode, u_ode1, u_ode2, semi)
        # Each RHS call consists of a `kick!` and a `drift!`.
        # First RHS call with the original coordinates
        TrixiParticles.kick!(dv_ode, v_ode, u_ode1, semi, 0.0)
        TrixiParticles.drift!(du_ode, v_ode, u_ode1, semi, 0.0)

        # Second RHS call with the perturbed coordinates
        TrixiParticles.kick!(dv_ode, v_ode, u_ode2, semi, 0.0)
        TrixiParticles.drift!(du_ode, v_ode, u_ode2, semi, 0.0)
    end

    result = @belapsed $double_rhs!($dv_ode, $du_ode, $v_ode, $u_ode1, $u_ode2, $semi)

    # Return average RHS time
    return 0.5 * result
end

function benchmark_wcsph_rhs_gpu(neighborhood_search, coordinates; parallel = true)
    density = 1000.0
    fluid = InitialCondition(; coordinates, density, mass = 0.1)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = PointNeighbors.search_radius(neighborhood_search)
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10.0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02, beta = 0.0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               density_diffusion = density_diffusion)

    semi = Semidiscretization(fluid_system; neighborhood_search)

    # TrixiParticles.jl creates a copy of the passed neighborhood search
    nhs = TrixiParticles.get_neighborhood_search(fluid_system, semi)
    initialize!(nhs, coordinates, coordinates)

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, nhs)

    semi_gpu = PointNeighbors.Adapt.adapt(CuArray, semi)

    v_ode = convert(CuArray, vec(vcat(fluid.velocity, fluid.density')))
    u_ode1 = convert(CuArray, vec(coordinates))
    dv_ode = zero(v_ode)
    du_ode = zero(u_ode1)

    # Similar to the alternating update, we simulate a real update in this benchmark
    # by alternating between two slightly different coordinate arrays.

    coordinates2 = copy(coordinates)
    # Perturb all coordinates with a perturbation factor of `0.015`.
    # This factor was tuned so that ~0.5% of the particles change their cell during an
    # update in 2D and ~0.7% in 3D.
    # These values are the same as the experimentally computed averages in 2D and 3D SPH
    # dam break simulations. So this benchmark replicates a real-life SPH update.
    perturb!(coordinates2, 0.015)

    u_ode2 = convert(CuArray, vec(coordinates2))

    function double_rhs!(dv_ode, du_ode, v_ode, u_ode1, u_ode2, semi)
        # Each RHS call consists of a `kick!` and a `drift!`.
        # First RHS call with the original coordinates
        TrixiParticles.kick!(dv_ode, v_ode, u_ode1, semi, 0.0)
        TrixiParticles.drift!(du_ode, v_ode, u_ode1, semi, 0.0)

        # Second RHS call with the perturbed coordinates
        TrixiParticles.kick!(dv_ode, v_ode, u_ode2, semi, 0.0)
        TrixiParticles.drift!(du_ode, v_ode, u_ode2, semi, 0.0)
    end

    result = @belapsed $double_rhs!($dv_ode, $du_ode, $v_ode, $u_ode1, $u_ode2, $semi_gpu)

    # Return average RHS time
    return 0.5 * result
end

function benchmark_wcsph_rhs_gpu32(neighborhood_search, coordinates_; parallel = true)
    coordinates = convert(Matrix{Float32}, coordinates_)
    density = 1000f0
    fluid = InitialCondition(; coordinates, density, mass = 0.1f0)

    # Compact support == smoothing length for the Wendland kernel
    smoothing_length = convert(Float32, PointNeighbors.search_radius(neighborhood_search))
    smoothing_kernel = WendlandC2Kernel{ndims(neighborhood_search)}()

    sound_speed = 10f0
    state_equation = StateEquationCole(; sound_speed, reference_density = density,
                                       exponent = 1)

    fluid_density_calculator = ContinuityDensity()
    viscosity = ArtificialViscosityMonaghan(alpha = 0.02f0, beta = 0f0)
    density_diffusion = DensityDiffusionMolteniColagrossi(delta = 0.1f0)

    fluid_system = WeaklyCompressibleSPHSystem(fluid, fluid_density_calculator,
                                               state_equation, smoothing_kernel,
                                               smoothing_length, viscosity = viscosity,
                                               acceleration = (0f0, 0f0, 0f0),
                                               density_diffusion = density_diffusion)

    semi = Semidiscretization(fluid_system; neighborhood_search)

    # TrixiParticles.jl creates a copy of the passed neighborhood search
    nhs = TrixiParticles.get_neighborhood_search(fluid_system, semi)
    initialize!(nhs, coordinates, coordinates)

    # Initialize the system
    TrixiParticles.initialize!(fluid_system, nhs)

    semi_gpu = PointNeighbors.Adapt.adapt(CuArray, semi)

    v_ode = cu(vec(vcat(fluid.velocity, fluid.density')))
    u_ode1 = cu(vec(coordinates))
    dv_ode = zero(v_ode)
    du_ode = zero(u_ode1)

    # Similar to the alternating update, we simulate a real update in this benchmark
    # by alternating between two slightly different coordinate arrays.

    coordinates2 = copy(coordinates)
    # Perturb all coordinates with a perturbation factor of `0.015`.
    # This factor was tuned so that ~0.5% of the particles change their cell during an
    # update in 2D and ~0.7% in 3D.
    # These values are the same as the experimentally computed averages in 2D and 3D SPH
    # dam break simulations. So this benchmark replicates a real-life SPH update.
    perturb!(coordinates2, 0.015)

    u_ode2 = cu(vec(coordinates2))

    function double_rhs!(dv_ode, du_ode, v_ode, u_ode1, u_ode2, semi)
        # Each RHS call consists of a `kick!` and a `drift!`.
        # First RHS call with the original coordinates
        TrixiParticles.kick!(dv_ode, v_ode, u_ode1, semi, 0f0)
        TrixiParticles.drift!(du_ode, v_ode, u_ode1, semi, 0f0)

        # Second RHS call with the perturbed coordinates
        TrixiParticles.kick!(dv_ode, v_ode, u_ode2, semi, 0f0)
        TrixiParticles.drift!(du_ode, v_ode, u_ode2, semi, 0f0)
    end

    result = @belapsed $double_rhs!($dv_ode, $du_ode, $v_ode, $u_ode1, $u_ode2, $semi_gpu)
    # return @belapsed $(TrixiParticles.interact!)($dv_ode, $v_ode, $u_ode1, $(semi_gpu.systems[1]), $(semi_gpu.systems[1]), $semi_gpu);

    # Return average RHS time
    return 0.5 * result
end
