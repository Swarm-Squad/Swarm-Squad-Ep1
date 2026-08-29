<div align="center">
<a href="https://github.com/Sang-Buster/Swarm-Squad"><img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/2d-sim/lib/img/banner.png" /></a>
<h1>Swarm Squad: Episode I – Surviving the Jam</h1>
<h6><small>A 3D web simulation platform combining behavior-based formation control with LLM-powered decision making for autonomous multi-agent systems.</small></h6>
<p><b>#Unmanned Aerial Vehicles &emsp; #Multi-agent Systems &emsp; #LLM Integration<br/>#3D Simulation &emsp; #Communication-aware &emsp; #Formation Control</b></p>
<p><small>Current <b>3D web</b> simulator. The original 2D PyQt app lives on the <a href="https://github.com/Swarm-Squad/Swarm-Squad-Ep1/tree/2d-sim"><code>2d-sim</code></a> branch.</small></p>
</div>

<h2 align="center">🔬 Research Evolution</h2>

This project builds upon our [previous research](https://github.com/speccoud/Swarm-Control) in formation control and swarm intelligence:

<img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/main/docs/img/gui.png" width="100%" />

- 🛸 **Low-Level Controller:** UAV agents with 3D formation control, path planning, and a V2V communication model<br/>
- 🤖 **High-Level Controller:** LLM chat agents that read live telemetry and issue strategic guidance through tools<br/>
- 🎯 **Goal:** Enable swarm resilience and mission completion under jamming, spoofing, and 3D obstacles

<h2 align="center">🚀 Getting Started</h2>

It is recommended to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create a virtual environment and install the following package.

```bash
uv pip install swarm-squad-ep1
```

To run the application, simply type:

```bash
swarm-squad-ep1
# or
swarm-squad-ep1 --help
```

Access points:

- GUI: `http://localhost:5000`
- Simulation API: `http://localhost:5001`

<div align="center">
  <h2>🛠️ Development Installation</h2>
</div>

1. **Clone the repository and navigate to project folder:**

   ```bash
   git clone https://github.com/Swarm-Squad/Swarm-Squad-Ep1
   cd Swarm-Squad-Ep1
   ```

2. **Install uv first:**

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   ```bash
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install the required packages:**
   **Option 1 (recommended):** Synchronizes environment with dependencies in pyproject.toml and uv.lock

   ```bash
   uv sync --extra dev
   source .venv/bin/activate # .venv\Scripts\activate for Windows
   cp .env.example .env
   ```

   **Option 2 (manual):** Manual editable installation without referencing lockfile

   ```bash
   uv venv --python 3.11 # Create virtual environment
   source .venv/bin/activate # .venv\Scripts\activate for Windows
   uv pip install -e ".[dev]"
   cp .env.example .env
   ```

4. **Start local dependencies** (Qdrant via Docker, plus Ollama for chat):

   ```bash
   docker compose up -d
   ollama serve
   ```

<div align="center">
  <h2>👨‍💻 Development Setup</h2>
</div>

1. **Install git hooks:**

   ```bash
   pre-commit install --install-hooks
   ```

   These hooks perform different checks at various stages:

   - `commit-msg`: Ensures commit messages follow the conventional format
   - `pre-commit`: Runs Ruff linting and formatting checks before each commit
   - `pre-push`: Performs final validation before pushing to remote

2. **Code Linting & Formatting:**

   ```bash
   ruff check --fix
   ruff check --select I --fix
   ruff format
   ```

3. **Run the application:**

   ```bash
   uv run swarm-squad-ep1
   ```

   Other useful commands:

   ```bash
   uv run swarm-squad-ep1 gui
   uv run swarm-squad-ep1 services --simulation-only
   uv run swarm-squad-ep1 services --chat-only
   uv run swarm-squad-ep1 research list
   uv run swarm-squad-ep1 research smoke
   uv run pytest -q
   ```

   Script-driven control:

   ```python
   from swarm_squad_ep1.client import SwarmSquadClient

   client = SwarmSquadClient()
   client.reset_simulation()
   client.set_algorithm(formation="communication_aware", path_algorithm="astar")
   client.add_jamming_zone(center=(12, 45, 10), radius=16, jam_type="low_jam")
   client.start_simulation()
   ```

<h2 align="center">📁 File Tree</h2>

```
📂Swarm-Squad-Ep1
 ┣ 📂docs                             // Guides, assignments, research notes
 ┣ 📂examples                         // Python client scripts and custom algorithms
 ┣ 📂src                              // Source code
 ┃ ┗ 📦swarm_squad_ep1                    // Python package
 ┃ ┃ ┣ 📂algo                                // Formation, path planning, attack/defense
 ┃ ┃ ┃ ┣ 📄formation.py                         // Formation control
 ┃ ┃ ┃ ┣ 📄path_planning_3d.py                  // 3D path planning
 ┃ ┃ ┃ ┣ 📄v2v_channel.py                       // Vehicle-to-vehicle channel model
 ┃ ┃ ┃ ┣ 📄jamming_response.py                  // Jamming response
 ┃ ┃ ┃ ┣ 📄spoofing.py                          // Spoofing attacks
 ┃ ┃ ┃ ┣ 📄crypto_auth.py                       // Authentication / integrity
 ┃ ┃ ┃ ┗ 📄llm_controller.py                    // LLM controller
 ┃ ┃ ┣ 📂chat                                // LLM chat service and tools
 ┃ ┃ ┣ 📂gui                                 // Browser 3D visualization
 ┃ ┃ ┃ ┗ 📂static
 ┃ ┃ ┃   ┣ 📄index.html                         // Web GUI
 ┃ ┃ ┃   ┗ 📂js
 ┃ ┃ ┃     ┗ 📄scene3d.js                       // 3D scene
 ┃ ┃ ┣ 📂rag                                 // Qdrant retrieval
 ┃ ┃ ┣ 📂research                            // Headless E1-E6 experiment harness
 ┃ ┃ ┣ 📂simulation                          // Simulation backend and API
 ┃ ┃ ┣ 📄cli.py                              // CLI entry point
 ┃ ┃ ┣ 📄client.py                           // Python client
 ┃ ┃ ┗ 📄runtime.py                          // Dual-service launcher
 ┣ 📂tests                            // API, algorithm, and frontend contract tests
 ┣ 📄docker-compose.yml               // Qdrant (+ optional Postgres)
 ┣ 📄.gitignore
 ┣ 📄.pre-commit-config.yaml
 ┣ 📄.python-version
 ┣ 📄README.md
 ┣ 📄pyproject.toml
 ┗ 📄uv.lock
```


Guides:

- Getting started: `docs/getting-started.md`
- Runtime and CLI: `docs/runtime-and-cli.md`
- Python client: `docs/client-api-reference.md`
- Algorithms and threat model: `docs/algorithms-and-threat-model.md`
- Research harness: `docs/research-harness.md`
