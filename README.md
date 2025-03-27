<div align="center">
<a href="https://github.com/Sang-Buster/Swarm-Squad"><img src="lib/imgbanner.png?raw=true" /></a>
<h1>Swarm Squad: Episode I – Surviving the Jam</h1>
<h6><small>A hybrid control architecture combining behavior-based formation control with LLM-powered decision making for autonomous multi-agent systems.</small></h6>
<p><b>#Unmanned Aerial Vehicles &emsp; #Multi-agent Systems &emsp; #LLM Integration<br/>#Behavior-based Control &emsp; #Communication-aware &emsp; #Formation Control</b></p>
</div>


<h2 align="center">🔬 Research Evolution</h2>

This project builds upon our previous research in formation control and swarm intelligence:

- 🚗 **Low-Level Controller:** Vehicle agents equipped with behavior-based and communication-aware formation control<br/>
- 🤖 **High-Level Controller:** LLM agents processing simulation data to provide strategic guidance<br/>
- 🎯 **Goal:** Enable swarm resilience and mission completion in challenging environments with jamming/obstacles

<h3 align="center">Supplementary Materials</h3>

<table>
  <tr>
    <th>Paper</th>
    <th>Presentation</th>
  </tr>
  <tr>
    <td align="center">
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Li-paper.pdf"><img src="lib/img/cover_paper.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Li-paper.pdf"><img src="https://img.shields.io/badge/View%20More-282c34?style=for-the-badge&logoColor=white" width="100" /></a>
    </td>
    <td align="center">
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="lib/img/cover_ppt.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="https://img.shields.io/badge/View%20Slides-282c34?style=for-the-badge&logoColor=white" /></a>   
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="lib/img/cover_video.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/assets/97267956/03072ecc-8218-40d9-a169-90774cb7c2ae"><img src="https://img.shields.io/badge/View%20Simulation%20Video-282c34?style=for-the-badge&logoColor=white" /></a>     
    </td>
  </tr>
</table>


<h2 align="center">🚀 Getting Started</h2>

Get [uv](https://docs.astral.sh/uv/getting-started/installation/) and create a virtual environment.
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install swarm-squad-ep1
```

To run the application, simply type:
```bash
swarm-squad
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

3. **Create a virtual environment:**
   ```bash
   uv venv
   ```

4. **Activate the virtual environment:**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   ```

   ```bash
   # Windows
   .venv\Scripts\activate
   ```

5. **Install the required packages:**
   ```bash
   uv pip install -e .
   ```

<div align="center">
  <h2>👨‍💻 Development Setup</h2>
</div>

1. **Install ruff and pre-commit:**
   ```bash
   uv pip install ruff pre-commit
   ```
   - `ruff` is a super fast Python linter and formatter.
   - `pre-commit` helps maintain code quality by running automated checks before commits are made.

2. **Install git hooks:**
   ```bash
   pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
   ```

   These hooks perform different checks at various stages:
   - `commit-msg`: Ensures commit messages follow the conventional format
   - `pre-commit`: Runs Ruff linting and formatting checks before each commit
   - `pre-push`: Performs final validation before pushing to remote
  
3. **Code Linting:**
   ```bash
   ruff check --fix
   ruff check --select I --fix
   ruff format
   ```

4. **Run the application:**
   ```bash
   uv run src/main.py
   ```


<h2 align="center">📁 File Tree</h2>

```
 📂Swarm-Squad-Ep1
 ┣ 📂lib                                  // Supplementary materials
 ┃ ┣ 📂img                                   // Readme Assets
 ┃ ┗ 📂old                                   // Original old code
 ┣ 📂src
 ┃ ┣ 📦Swarm-Squad-Ep1                   // Source Code
 ┃ ┃ ┣ 📄main.py                            // Entry point
 ┃ ┃ ┣ 📄config.py                          // Configuration parameters
 ┃ ┃ ┣ 📄utils.py                           // Core utility functions
 ┃ ┃ ┣ 📄visualization.py                   // Visualization functions
 ┃ ┃ ┣ 📂models                         // Model components
 ┃ ┃ ┃ ┗ 📄swarm_state.py                   // Swarm state management
 ┃ ┃ ┣ 📂controllers                    // Controllers for swarm behavior
 ┃ ┃ ┃ ┣ 📄base_controller.py               // Base controller interface
 ┃ ┃ ┃ ┣ 📄communication_controller.py      // Communication-aware controller
 ┃ ┃ ┃ ┣ 📄behavior_controller.py           // Behavior-based controller
 ┃ ┃ ┃ ┣ 📄llm_controller.py                // LLM controller
 ┃ ┃ ┃ ┗ 📄controller_factory.py            // Controller management system
 ┃ ┃ ┣ 📂gui                            // GUI components
 ┃ ┃ ┗ ┗ 📄formation_control_gui.py         // GUI application
 ┣ 📄.gitignore
 ┣ 📄.pre-commit-msg_template.yaml
 ┣ 📄.pre-commit-config.yaml
 ┣ 📄.python-version
 ┣ 📄LICENSE
 ┣ 📄pyproject.toml
 ┣ 📄README.md
 ┗ 📄uv.lock
```
