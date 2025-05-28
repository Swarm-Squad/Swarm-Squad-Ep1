<div align="center">
<a href="https://github.com/Sang-Buster/Swarm-Squad"><img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/main/lib/img/banner.png" /></a>
<h1>Swarm Squad: Episode I – Surviving the Jam</h1>
<h6><small>A hybrid control architecture combining behavior-based formation control with LLM-powered decision making for autonomous multi-agent systems.</small></h6>
<p><b>#Unmanned Aerial Vehicles &emsp; #Multi-agent Systems &emsp; #LLM Integration<br/>#Behavior-based Control &emsp; #Communication-aware &emsp; #Formation Control</b></p>
</div>

<h2 align="center">🔬 Research Evolution</h2>

This project builds upon our previous research in formation control and swarm intelligence:

<img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/main/lib/img/gui.png" width="100%" />

- 🚗 **Low-Level Controller:** Vehicle agents equipped with behavior-based and communication-aware formation control<br/>
- 🤖 **High-Level Controller:** LLM agents processing simulation data to provide strategic guidance<br/>
- 🎯 **Goal:** Enable swarm resilience and mission completion in challenging environments with jamming/obstacles

<!-- <h3 align="center">Supplementary Materials</h3>

<table>
  <tr>
    <th>Paper</th>
    <th>Presentation</th>
  </tr>
  <tr>
    <td align="center">
          <a href="https://github.com/Swarm-Squad/Swarm-Squad-Ep1/blob/main/lib/Xing-paper.pdf"><img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/main/lib/img/cover_paper.png" /></a>
          <a href="https://github.com/Swarm-Squad/Swarm-Squad-Ep1/blob/main/lib/Xing-paper.pdf"><img src="https://img.shields.io/badge/View%20More-282c34?style=for-the-badge&logoColor=white" width="100" /></a>
    </td>
    <td align="center">
          <a href="https://github.com/Swarm-Squad/Swarm-Squad-Ep1/blob/main/lib/Xing-ppt.pdf"><img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/main/lib/img/cover_ppt.png" /></a>
          <a href="https://github.com/Swarm-Squad/Swarm-Squad-Ep1/blob/main/lib/Xing-ppt.pdf"><img src="https://img.shields.io/badge/View%20Slides-282c34?style=for-the-badge&logoColor=white" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/assets/97267956/03072ecc-8218-40d9-a169-90774cb7c2ae"><img src="https://raw.githubusercontent.com/Swarm-Squad/Swarm-Squad-Ep1/refs/heads/main/lib/img/cover_video.png" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/assets/97267956/03072ecc-8218-40d9-a169-90774cb7c2ae"><img src="https://img.shields.io/badge/View%20Simulation%20Video-282c34?style=for-the-badge&logoColor=white" /></a>
    </td>
  </tr>
</table> -->

<h2 align="center">🚀 Getting Started</h2>

It is recommended to use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create a virtual environment and install the following package.

```bash
pip install swarm-squad-ep1
```

To run the application, simply type:

```bash
swarm-squad-ep1
# or
swarm-squad-ep1 --help
```

<div align="center">
  <h2>🛠️ Development Installation</h2>
</div>

1. **Clone the repository and navigate to project folder:**

   ```bash
   git clone https://github.com/Sang-Buster/Swarm-Squad-Ep1
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
   uv sync
   source .venv/bin/activate # .venv\Scripts\activate for Windows
   ```

   **Option 2 (manual):** Manual editable installation without referencing lockfile

   ```bash
   uv venv --python 3.10 # Create virtual environment
   source .venv/bin/activate # .venv\Scripts\activate for Windows
   uv pip install -e .
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
   uv run src/swarm_squad/main.py
   ```

<h2 align="center">📁 File Tree</h2>

```
📂Swarm-Squad-Ep1
 ┣ 📂lib                              // Supplementary materials
 ┃ ┣ 📂img                                // Readme Assets
 ┃ ┣ 📂old                                // Original old code
 ┃ ┣ 📄demo.mp4                           // Demo Video
 ┃ ┣ 📄paper.pdf                          // Paper
 ┃ ┗ 📄ppt.pdf                            // Presentation
 ┣ 📂logs                             // SimulationLogs
 ┣ 📂src                              // Source Code
 ┃ ┗ 📦swarm_squad                        // Python package
 ┃ ┃ ┣ 📂controllers                         // Controllers for swarm behavior
 ┃ ┃ ┃ ┣ 📄base_controller.py                   // Base controller interface
 ┃ ┃ ┃ ┣ 📄behavior_controller.py               // Behavior-based controller
 ┃ ┃ ┃ ┣ 📄controller_factory.py                // Controller management system
 ┃ ┃ ┃ ┣ 📄formation_controller.py              // Formation control
 ┃ ┃ ┃ ┗ 📄llm_controller.py                    // LLM controller
 ┃ ┃ ┣ 📂gui                                 // GUI components
 ┃ ┃ ┃ ┗ 📄formation_control_gui.py             // GUI application
 ┃ ┃ ┣ 📂models                              // Model components
 ┃ ┃ ┃ ┗ 📄swarm_state.py                       // Swarm state management
 ┃ ┃ ┣ 📂tests                               // Test cases
 ┃ ┃ ┃ ┗ 📄test_ollama.py                       // Test for Ollama
 ┃ ┃ ┣ 📄config.py                           // Configuration parameters
 ┃ ┃ ┣ 📄main.py                             // Entry point
 ┃ ┃ ┣ 📄utils.py                            // Core utility functions
 ┃ ┃ ┗ 📄visualization.py                    // Visualization functions
 ┣ 📄.gitignore
 ┣ 📄.pre-commit-config.yaml
 ┣ 📄.python-version
 ┣ 📄LICENSE
 ┣ 📄README.md
 ┣ 📄pyproject.toml
 ┗ 📄uv.lock
```
