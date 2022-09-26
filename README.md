# Using "shaped controls" in GRAPE

See discussion at https://github.com/JuliaQuantumControl/QuantumControl.jl/issues/15

## Prerequisites

* Installation of Jupyter with [jupytext](https://github.com/mwouts/jupytext) plugin
* Julia 1.8 with [DrWatson](https://github.com/JuliaDynamics/DrWatson.jl) and [IJulia](https://github.com/JuliaLang/IJulia.jl)
* Ensure that the Julia [kernel specification](https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernelspecs) is set up correctly. The `kernel.json` file should look something like this:

  ```
  {
    "display_name": "Julia 1.8.2",
    "argv": [
      "/Applications/Julia-1.8.app/Contents/Resources/julia/bin/julia",
      "-i",
      "--color=yes",
      "--project=@.",
      "/Users/goerz/.julia/packages/IJulia/e8kqU/src/kernel.jl",
      "{connection_file}"
    ],
    "language": "julia",
    "env": {},
    "interrupt_mode": "signal"
  }
  ```

  Note the `--project=@.` argument

## Usage

* Instantiate the environment:

  ```
  julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

* Run a jupyter server

* Assuming the `Jupytext` plugin is installed and set up correctly, open either `shaped_control.jl` or `shaped_control.ipynb`. Without `Jupytext`, use the `.ipynb` file.
