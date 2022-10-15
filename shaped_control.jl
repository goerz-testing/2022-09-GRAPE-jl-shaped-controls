# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Julia 1.8 (4 threads)
#     language: julia
#     name: julia-1.8-multithread
# ---

# # Quantum gates with "shaped controls"

# $
# \newcommand{tr}[0]{\operatorname{tr}}
# \newcommand{diag}[0]{\operatorname{diag}}
# \newcommand{abs}[0]{\operatorname{abs}}
# \newcommand{pop}[0]{\operatorname{pop}}
# \newcommand{aux}[0]{\text{aux}}
# \newcommand{opt}[0]{\text{opt}}
# \newcommand{tgt}[0]{\text{tgt}}
# \newcommand{init}[0]{\text{init}}
# \newcommand{lab}[0]{\text{lab}}
# \newcommand{rwa}[0]{\text{rwa}}
# \newcommand{bra}[1]{\langle#1\vert}
# \newcommand{ket}[1]{\vert#1\rangle}
# \newcommand{Bra}[1]{\left\langle#1\right\vert}
# \newcommand{Ket}[1]{\left\vert#1\right\rangle}
# \newcommand{Braket}[2]{\left\langle #1\vphantom{#2}\mid{#2}\vphantom{#1}\right\rangle}
# \newcommand{op}[1]{\hat{#1}}
# \newcommand{Op}[1]{\hat{#1}}
# \newcommand{dd}[0]{\,\text{d}}
# \newcommand{Liouville}[0]{\mathcal{L}}
# \newcommand{DynMap}[0]{\mathcal{E}}
# \newcommand{identity}[0]{\mathbf{1}}
# \newcommand{Norm}[1]{\lVert#1\rVert}
# \newcommand{Abs}[1]{\left\vert#1\right\vert}
# \newcommand{avg}[1]{\langle#1\rangle}
# \newcommand{Avg}[1]{\left\langle#1\right\rangle}
# \newcommand{AbsSq}[1]{\left\vert#1\right\vert^2}
# \newcommand{Re}[0]{\operatorname{Re}}
# \newcommand{Im}[0]{\operatorname{Im}}
# $

using DrWatson
@quickactivate "GRAPETests"
using QuantumControl

# Here, we explore the use of "shaped pulses" in a GRAPE optimization. The physical drive corresponds to the control amplitudes $\Omega(t) = S(t) \epsilon(t)$, where $\epsilon(t)$ is the control function to be optimized via GRAPE. The shape function $S(t)$ is not affected by the optimization. Choosing $S(t)$ to smoothly switch on from zero at $t=0$ and back down to zero at $t=T$ ensures that the optimized controls (and the resulting amplitudes $\Omega(t)$) maintain the same boundary conditions of zero at the beginning and end of the time grid. The role of $S(t)$ here is very similar to the "update shape" $S(t)$ used in Krotov'e method.
#
# The example is adapted from the [GRAPE Example 2](https://juliaquantumcontrol.github.io/GRAPE.jl/stable/examples/perfect_entanglers/).

# ## Shaped Control Amplitudes Implementation

# +
import QuantumPropagators.Generators:
    substitute_controls, getcontrols, evalcontrols, getcontrolderiv


"""Control amplitude ``aₗ(ϵₗ(t)) = S(t) ϵ(t)``."""
struct ShapedControlAmplitude
    control  # ϵ(t)
    shape  # S(t)
end

"""Amplitude S(t) that is not a control."""
struct StaticShapeAmplitude
    shape
end


# for plotting
(Ω::ShapedControlAmplitude)(t::Float64) = Ω.control(t) * Ω.shape(t)


substitute_controls(ampl::ShapedControlAmplitude, controls_map) =
    ShapedControlAmplitude(get(controls_map, ampl.control, ampl.control), ampl.shape)

getcontrols(ampl::ShapedControlAmplitude) = (ampl.control,)


# Midpoint of n'th interval of tlist, but snap to beginning/end (that's
# because any S(t) is likely exactly zero at the beginning and end, and we
# want to use that value for the first and last time interval)
function _t(tlist, n)
    @assert 1 <= n <= (length(tlist) - 1)  # n is an *interval* of `tlist`
    if n == 1
        t = tlist[begin]
    elseif n == length(tlist) - 1
        t = tlist[end]
    else
        dt = tlist[n+1] - tlist[n]
        t = tlist[n] + dt / 2
    end
    return t
end


evalcontrols(ampl::ShapedControlAmplitude, vals_dict, tlist, n) =
    ampl.shape(_t(tlist, n)) * vals_dict[ampl.control]

evalcontrols(ampl::StaticShapeAmplitude, vals_dict, tlist, n) = ampl.shape(_t(tlist, n))

getcontrolderiv(ampl::ShapedControlAmplitude, control) =
    (control ≡ ampl.control) ? StaticShapeAmplitude(ampl.shape) : 0.0
# -

# ## Hamiltonian and guess pulses

# We will write the Hamiltonian in units of GHz (angular frequency; the factor
# 2π is implicit) and ns:

const GHz = 2π
const MHz = 0.001GHz
const ns = 1.0
const μs = 1000ns;

# The Hamiltonian and parameter are taken from Goerz et. al., Phys. Rev. A 91,
# 062307 (2015)., cf. Table 1 in that Reference.

# +
⊗ = kron
const 𝕚 = 1im
const N = 6  # levels per transmon

using LinearAlgebra
using SparseArrays

function tm_hamiltonian(;
    Ωre,
    Ωim,
    N=N,  # levels per transmon
    ω₁=4.380GHz,
    ω₂=4.614GHz,
    ωd=4.498GHz,
    α₁=-210MHz,
    α₂=-215MHz,
    J=-3MHz,
    λ=1.03,
    use_sparse=:auto
)
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, N, N))
    b̂₁ = spdiagm(1 => complex.(sqrt.(collect(1:N-1)))) ⊗ 𝟙
    b̂₂ = 𝟙 ⊗ spdiagm(1 => complex.(sqrt.(collect(1:N-1))))
    b̂₁⁺ = sparse(b̂₁')
    b̂₂⁺ = sparse(b̂₂')
    n̂₁ = sparse(b̂₁' * b̂₁)
    n̂₂ = sparse(b̂₂' * b̂₂)
    n̂₁² = sparse(n̂₁ * n̂₁)
    n̂₂² = sparse(n̂₂ * n̂₂)
    b̂₁⁺_b̂₂ = sparse(b̂₁' * b̂₂)
    b̂₁_b̂₂⁺ = sparse(b̂₁ * b̂₂')

    ω̃₁ = ω₁ - ωd
    ω̃₂ = ω₂ - ωd

    Ĥ₀ = sparse(
        (ω̃₁ - α₁ / 2) * n̂₁ +
        (α₁ / 2) * n̂₁² +
        (ω̃₂ - α₂ / 2) * n̂₂ +
        (α₂ / 2) * n̂₂² +
        J * (b̂₁⁺_b̂₂ + b̂₁_b̂₂⁺)
    )

    Ĥ₁re = (1 / 2) * (b̂₁ + b̂₁⁺ + λ * b̂₂ + λ * b̂₂⁺)
    Ĥ₁im = (𝕚 / 2) * (b̂₁⁺ - b̂₁ + λ * b̂₂⁺ - λ * b̂₂)

    if ((N < 5) && (use_sparse ≢ true)) || use_sparse ≡ false
        H = hamiltonian(Array(Ĥ₀), (Array(Ĥ₁re), Ωre), (Array(Ĥ₁im), Ωim))
    else
        H = hamiltonian(Ĥ₀, (Ĥ₁re, Ωre), (Ĥ₁im, Ωim))
    end
    return H

end;
# -

# We choose a pulse duration of 400 ns. The guess pulse amplitude is 35 MHz,
# with a 15 ns switch-on/-off time. The Hamiltonian is written in a rotating
# frame, so in general, the control field is allowed to be complex-valued. We
# separate this into two control fields, one for the real part and one for the
# imaginary part. Initially, the imaginary part is zero, corresponding to a
# field exactly at the frequency of the rotating frame.

# +
using QuantumControl.Shapes: flattop

function guess_pulses(; T=400ns, E₀=35MHz, dt=0.1ns, t_rise=15ns)

    tlist = collect(range(0, T, step=dt))
    Ωre = ShapedControlAmplitude(
        t -> E₀ * flattop(t, T=T, t_rise=t_rise), # ϵ₁(t)
        t -> flattop(t, T=T, t_rise=t_rise)       # S₁(t)
    )
    Ωim = ShapedControlAmplitude(
        t -> 0.0,                                 # ϵ₂(t)
        t -> flattop(t, T=T, t_rise=t_rise)       # S₂(t)
    )
    return tlist, Ωre, Ωim

end

tlist, Ωre_guess, Ωim_guess = guess_pulses();
# -

# We can visualize this:

# +
using Plots
Plots.default(
    linewidth               = 3,
    size                    = (550, 300),
    legend                  = :right,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.8),
)

function plot_complex_pulse(tlist, Ω; time_unit=:ns, ampl_unit=:MHz, kwargs...)

    ax1 = plot(
        tlist ./ eval(time_unit),
        abs.(Ω) ./ eval(ampl_unit);
        label="|Ω|",
        xlabel="time ($time_unit)",
        ylabel="amplitude ($ampl_unit)",
        kwargs...
    )

    ax2 = plot(
        tlist ./ eval(time_unit),
        angle.(Ω) ./ π;
        label="ϕ(Ω)",
        xlabel="time ($time_unit)",
        ylabel="phase (π)"
    )

    plot(ax1, ax2, layout=(2, 1))

end

plot_complex_pulse(tlist, Ωre_guess.(tlist) + 𝕚 * Ωim_guess.(tlist))
# -

# We now instantiate the Hamiltonian with these control fields:

H = tm_hamiltonian(Ωre=Ωre_guess, Ωim=Ωim_guess);

typeof(H)

# ## Logical basis for two-qubit gates

# For simplicity, we will be define the qubits in the *bare* basis, i.e.
# ignoring the static coupling $J$.

# +
function ket(i::Int64; N=N)
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

function ket(indices::Int64...; N=N)
    Ψ = ket(indices[1]; N=N)
    for i in indices[2:end]
        Ψ = Ψ ⊗ ket(i; N=N)
    end
    return Ψ
end

function ket(label::AbstractString; N=N)
    indices = [parse(Int64, digit) for digit in label]
    return ket(indices...; N=N)
end;
# -

basis = [ket("00"), ket("01"), ket("10"), ket("11")];

# ## Optimizing for a specific quantum gate

# Our target gate is $\Op{O} = \sqrt{\text{iSWAP}}$:

SQRTISWAP = [
    1  0    0   0
    0 1/√2 𝕚/√2 0
    0 𝕚/√2 1/√2 0
    0  0    0   1
];

# For each basis state, we get a target state that results from applying the
# gate to the basis state (you can convince yourself that this equivalent
# multiplying the transpose of the above gate matrix to the vector of basis
# states):

basis_tgt = transpose(SQRTISWAP) * basis;

# The mapping from each initial (basis) state to the corresponding target state
# constitutes an "objective" for the optimization:

objectives = [
    Objective(initial_state=Ψ, target_state=Ψtgt, generator=H) for
    (Ψ, Ψtgt) ∈ zip(basis, basis_tgt)
];

# We can analyze how all of the basis states evolve under the guess controls in
# one go:

# +
using QuantumControl: propagate_objectives

guess_states = propagate_objectives(objectives, tlist; use_threads=true);
# -

# The gate implemented by the guess controls is

U_guess = [basis[i] ⋅ guess_states[j] for i = 1:4, j = 1:4];

# We will optimize these objectives with a square-modulus functional

using QuantumControl.Functionals: J_T_sm

# The initial value of the functional is

J_T_sm(guess_states, objectives)

# which is the gate error

1 - (abs(tr(U_guess' * SQRTISWAP)) / 4)^2

# Now, we define the full optimization problems on top of the list of
# objectives, and with the optimization functional:

problem = ControlProblem(
    objectives=objectives,
    tlist=tlist,
    iter_stop=100,
    J_T=J_T_sm,
    check_convergence=res -> begin
        (
            (res.J_T > res.J_T_prev) &&
            (res.converged = true) &&
            (res.message = "Loss of monotonic convergence")
        )
        ((res.J_T <= 1e-2) && (res.converged = true) && (res.message = "J_T < 10⁻²"))
    end,
    use_threads=true,
);

opt_result = @optimize_or_load(
    datadir("GATE_OCT_shaped.jld2"),
    problem;
    method=:GRAPE,
    force=true
);

opt_result

# We extract the optimized control field from the optimization result and plot
# it

ϵ_opt = opt_result.optimized_controls[1] + 𝕚 * opt_result.optimized_controls[2]
plot_complex_pulse(tlist, ϵ_opt)

# +
ϵ_opt_func = t -> begin
    n = min(searchsortedfirst(tlist, t), length(tlist))
    return ϵ_opt[n]
end
Ω_opt = ShapedControlAmplitude(ϵ_opt_func, Ωre_guess.shape)

plot_complex_pulse(tlist, Ω_opt.(tlist))
# -

typeof(Ω_opt)

# We then propagate the optimized control field to analyze the resulting
# quantum gate:

opt_states = propagate_objectives(
    objectives,
    tlist;
    use_threads=true,
    controls_map=IdDict(
        Ωre_guess.control => opt_result.optimized_controls[1],
        Ωim_guess.control => opt_result.optimized_controls[2]
    )
);

# The resulting gate is

U_opt = [basis[i] ⋅ opt_states[j] for i = 1:4, j = 1:4];

# and we can verify the resulting fidelity

(abs(tr(U_opt' * SQRTISWAP)) / 4)^2

# ---
#
# *This notebook was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
