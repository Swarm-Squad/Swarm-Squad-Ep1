<div align="center">
<a href="https://github.com/Sang-Buster/Swarm-Squad"><img src="img/banner.png?raw=true" /></a>
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
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Li-paper.pdf"><img src="img/cover_paper.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Li-paper.pdf"><img src="https://img.shields.io/badge/View%20More-282c34?style=for-the-badge&logoColor=white" width="100" /></a>
    </td>
    <td align="center">
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="img/cover_ppt.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="https://img.shields.io/badge/View%20Slides-282c34?style=for-the-badge&logoColor=white" /></a>   
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/blob/main/lib/Xing-ppt.pdf"><img src="img/cover_video.png?raw=true" /></a>
          <a href="https://github.com/Sang-Buster/Communication-aware-Formation-Control/assets/97267956/03072ecc-8218-40d9-a169-90774cb7c2ae"><img src="https://img.shields.io/badge/View%20Simulation%20Video-282c34?style=for-the-badge&logoColor=white" /></a>     
    </td>
  </tr>
</table>



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

3. **Install tkinter (if not already installed):**
   ```bash
   # Debian/Ubuntu
   sudo apt-get install python3-tk

   # Arch Linux
   sudo pacman -Syu tk

   # Fedora
   sudo dnf install python3-tkinter

   # macOS (Homebrew)
   brew install python-tk

   # After installing, check which Python version has tkinter:
   for v in $(ls /usr/bin/python3*); do
       echo -n "$v: "
       $v -c "import tkinter; print('tkinter available')" 2>/dev/null || echo "no tkinter"
   done
   ```

4. **Create a virtual environment using the Python version that has tkinter:**
   ```bash
   # Replace X.Y with the version number found above (e.g., 3.13, 3.12, 3.11)
   uv venv --python 3.X
   ```

5. **Activate the virtual environment:**
   ```bash
   # macOS/Linux
   source .venv/bin/activate
   ```

   ```bash
   # Windows
   .venv\Scripts\activate
   ```

6. **Install the required packages:**
   ```bash
   uv pip install -r requirements.txt
   ```

<div align="center">
  <h2>👨‍💻 Development Setup</h2>
</div>

1. **Install pre-commit:**
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
   ruff check
   ruff format
   ```

4. **Run the application:**
   ```bash
   python src/main.py
   ```


<h2 align="center">📁 File Tree</h2>

```text
📦Swarm-Squad-Ep1
 ┣ 📂img                              // Readme Assets
 ┣ 📂lib                              // Supplementary Materials
 ┣ 📂src                              // Source Code
 ┃ ┃ ┣ 📄main.py
 ┃ ┃ ┗ 📄utils.py
 ┣ 📄.gitignore
 ┗ 📄README.md
```
