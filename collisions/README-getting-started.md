## Getting Started With The Julia Script

This guide provides instructions on how to set up the Julia environment for the charger collision model. The script uses specific versions of packages, which are listed in `Project.toml` and `Manifest.toml`.

Follow these steps to set up your environment:

### Install Julia

Ensure that you have Julia installed on your system. You can download it from [the Julia website](https://julialang.org/downloads/).

### Clone the Repository

Clone this repository to your local machine:
```
git clone git@github.com:EECA-NZ/ev-charger-analysis.git
```

### Open Julia REPL

Open Julia's REPL (Read-Eval-Print Loop), which is the Julia command line interface.

### Navigate to the Project Directory

In the Julia REPL, change your current working directory to the project directory using the `cd` command:

```julia
cd("path/to/the/project")  # Replace with the path to your project directory
```

### Activate the Project
Activate the project environment. This tells Julia to use the specific package versions listed in Project.toml and Manifest.toml.
```julia
using Pkg
Pkg.activate(".")
```

### Install the Packages
Install all the necessary dependencies by running:
```
Pkg.instantiate()
```

To verify that the correct packages have been installed, you can check the status of the installed packages: `Pkg.status()`. Assuming all is well, the script should now run if you run
```
include("single-charger-collision-model.jl")
```