<h1 align="center">Swarm Squad: Episode I – Surviving the Jam</h1>

<h6 align="center"><small>A hybrid control architecture combining behavior-based formation control with LLM-powered decision making for autonomous multi-agent systems.</small></h6>

<p align="center"><b>#Unmanned Aerial Vehicles &emsp; #Multi-agent Systems &emsp; #LLM Integration<br/>#Behavior-based Control &emsp; #Communication-aware &emsp; #Formation Control</b></p>

<p>
🚗 <b>Low-Level Controller:</b> Vehicle agents equipped with behavior-based and communication-aware formation control<br/>
🤖 <b>High-Level Controller:</b> LLM agents processing simulation data to provide strategic guidance<br/>
🎯 <b>Goal:</b> Enable swarm resilience and mission completion in challenging environments with jamming/obstacles
</p>

<div align="center">
  <h2>🛠️ Setup & Installation</h2>
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

3. **Create a virtual environment at `/weather-dashboard/.venv/`:**
   ```bash
   uv venv --python 3.12.1
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
   uv pip install -r requirements.txt
   ```

<div align="center">
  <h2>👨‍💻 Development Setup</h2>
</div>

### Development Instructions 

1. **Install pre-commit:**
   ```bash
   uv pip install pre-commit
   ```
   Pre-commit helps maintain code quality by running automated checks before commits are made.

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
   ruff check
   ruff format
   ```

4. **Run the application:**
   ```bash
   python src/app.py
   ```

<h2 align="center">File Tree</h2>

```text
📦
 ┣ 📂img                              // Readme Assets
 ┣ 📂lib                              // Supplementary Materials
 ┣ 📂src                              // Source Code
 ┃ ┃ ┣ 📄main.py
 ┃ ┃ ┗ 📄utils.py
 ┣ 📄.gitignore
 ┗ 📄README.md
```

<h2 align="center">Supplementary Materials</h2>

<table>
  <tr>
    <th>Paper</th>
    <th>Presentation</th>
  </tr>
  <tr>
    <td align="center">
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Li-paper.pdf"><img src="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/img/cover_paper.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Li-paper.pdf"><img src="https://img.shields.io/badge/View%20More-282c34?style=for-the-badge&logoColor=white" width="100" /></a>
    </td>
    <td align="center">
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/img/cover_ppt.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="https://img.shields.io/badge/View%20Slides-282c34?style=for-the-badge&logoColor=white" /></a>   
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/img/cover_ppt.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/assets/97267956/03072ecc-8218-40d9-a169-90774cb7c2ae"><img src="https://img.shields.io/badge/View%20Simulation%20Video-282c34?style=for-the-badge&logoColor=white" /></a>     
    </td>
  </tr>
</table>
