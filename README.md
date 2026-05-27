# FloodMIND Nowcast Blend

Containerised adaptation of the nowcast blending workflow developed by Joep Bosdijk: https://github.com/Joep1999/DestinE_code/tree/nowcast_blend

The workflow and code structure have been kept as close as possible to the original implementation while adding support for configuration, reproducibility and deployment. Validation against the original workflow is ongoing.

## Infrastructure Setup

A `Dockerfile` is provided to build an image containing:

- application code
- runtime configuration
- static resources under `resources/`
- the custom `pysteps_destine` package

The image does not contain runtime data, credentials, or the DGMR model.

Use `docker-compose.infra.example.yml` as an example of the runtime contract for infrastructure.

### Runtime Credentials

The pipeline needs these credentials as environment variables:

- `KNMI_API_KEY`: KNMI Data Platform API key for radar files
- `POLYTOPE_USER_KEY`: Destination Earth / Polytope key for DestinE Extremes DT files

Provide these through Secret objects. Do not bake them into the image or config files.

### DGMR Model

If `DGMR_MODEL_PATH` is set, the application loads the DGMR TensorFlow model from that path.

If `DGMR_MODEL_PATH` is unset, the application loads the model from the remote TensorFlow Hub / Google Cloud Storage path at runtime. This requires outbound network access and is slower. For operational use, I suggest mounting a pre-downloaded model using

```bash
DGMR_MODEL_PATH=/model
```

with:

```yaml
volumes:
  - ./model:/model:ro
```

### Runtime Storage

The application writes runtime data under `DATA_DIR`.
This includes:

- downloaded input radar and DestinE files
- preprocessed input DestinE NetCDF files
- cached DGMR nowcasts
- blended outputs
- Hydra logs and resolved run configs

Without persistent storage, every run should expect to redownload/recompute missing inputs and intermediates. That is possible, but slower and harder to debug.

Output and logs location, to copy to persistent storage?, is:

- final blended NetCDF files: `/data/output/blended/**/*.nc`
- run log: `/data/logs/hydra/<run>/main.log`

### Runtime Requirements

Suggested starting point for an infra run that generates DGMR:

- Docker image size: about 5 GiB
- CPU: 10 vCPU recommended
- memory: 16 GiB minimum, 20-24 GiB safer
- disk: _at least_ 10 GiB for image plus runtime outputs/intermediates
- `DGMR_MAX_WORKERS=1` unless the runtime has enough memory for parallel DGMR members

Network access:

- KNMI Data Platform for radar downloads
- Destination Earth / Polytope for DestinE downloads
- TensorFlow Hub / Google Cloud Storage if `DGMR_MODEL_PATH` is unset

Real-time deployment should run:

```bash
python -m nowcast_blend.main settings.rt=true
```

## Local Development

### Environment Variables

Create a local `.envrc` from the example:

```bash
cp .envrc.example .envrc
```

Edit `.envrc` and set your local values. At minimum, local runs need:

```bash
export KNMI_API_KEY="..."
export POLYTOPE_USER_KEY="..."
```

For local Python runs, `.envrc` can also set local paths such as `DATA_DIR` and `DGMR_MODEL_PATH`.

For Docker compose, non-secret runtime paths are already set in `docker-compose.yml`; compose only needs the secrets to be exported in your shell.

Activate the variables:

```bash
source .envrc
```

### Python Environment

On macOS, install and export the compiler/OpenMP dependencies:

```bash
brew install libomp llvm
export CC="$(brew --prefix llvm)/bin/clang"
export CXX="$(brew --prefix llvm)/bin/clang++"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib"
```

Create a local Python environment:

```bash
pyenv local 3.10.14
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --timeout 1000 --retries 10
```


### Run Locally With Python

Scientific run settings are in `configs/nowcast-blend.yaml`.

Run:

```bash
source .envrc
python -m nowcast_blend.main
```

### Run Locally With Docker Compose

Build:

```bash
docker compose build --no-cache
```

Run:

```bash
source .envrc
docker compose run --rm nowcast-blend
```

The local compose file mounts:

- `./data` to `/data`
- `./model` to `/model`

The local compose command currently overrides the config to use a smaller ensemble (`n_ens_members_dgmr` and `n_ens_members`) for Docker testing. Remove those overrides in `docker-compose.yml` if you want the full config values.

### Recreate Local Python Environment

```bash
deactivate
rm -rf .venv
```

Then repeat the Python environment setup.
