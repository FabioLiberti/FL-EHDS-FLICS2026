# FL-EHDS Project Structure
## Modular Development for FLICS 2026 Submission

---

## 📁 Repository Structure

```
FL-EHDS-FLICS2026/
│
├── 📄 README.md                          # Project overview
├── 📄 CHANGELOG.md                       # Version history (GitVersion)
├── 📄 .gitversion.yml                    # GitVersion configuration
├── 📄 .gitignore                         # Git ignore rules
│
├── 📁 paper/                             # Main paper (Overleaf sync)
│   ├── 📄 main.tex                       # Master LaTeX file
│   ├── 📄 abstract.tex                   # Abstract (150 words)
│   ├── 📄 introduction.tex               # Section 1
│   ├── 📄 background.tex                 # Section 2
│   ├── 📄 framework.tex                  # Section 3 (main contribution)
│   ├── 📄 evidence.tex                   # Section 4
│   ├── 📄 roadmap.tex                    # Section 5
│   ├── 📄 discussion.tex                 # Section 6
│   ├── 📄 references.bib                 # BibTeX references
│   └── 📁 figures/
│       ├── 📄 fig1-fl-workflow.pdf       # Figure 1
│       ├── 📄 fig2-fl-ehds-arch.pdf      # Figure 2 (main)
│       └── 📄 fig2-fl-ehds-arch.drawio   # Source for editing
│
├── 📁 src/                               # Source materials
│   ├── 📁 slr-data/                      # Original SLR data
│   │   ├── 📄 included-studies.csv       # 47 included documents
│   │   ├── 📄 extraction-form.xlsx       # Data extraction
│   │   ├── 📄 quality-assessment.xlsx    # MMAT scores
│   │   └── 📄 prisma-flow.xlsx           # PRISMA numbers
│   │
│   ├── 📁 framework/                     # Framework specifications
│   │   ├── 📄 architecture.md            # Detailed architecture
│   │   ├── 📄 components.md              # Component descriptions
│   │   ├── 📄 compliance-checklist.md    # Compliance checkpoints
│   │   └── 📄 metrics.md                 # Evaluation metrics
│   │
│   └── 📁 evidence/                      # Evidence synthesis
│       ├── 📄 barrier-taxonomy.md        # Technical barriers
│       ├── 📄 legal-uncertainties.md     # Legal issues
│       └── 📄 organizational-barriers.md # Org barriers
│
├── 📁 figures/                           # Figure source files
│   ├── 📁 drawio/                        # Draw.io sources
│   │   ├── 📄 fl-workflow.drawio
│   │   └── 📄 fl-ehds-architecture.drawio
│   ├── 📁 tikz/                          # TikZ sources (LaTeX)
│   │   ├── 📄 timeline.tex
│   │   └── 📄 layers.tex
│   └── 📁 exports/                       # PDF exports for paper
│
├── 📁 docs/                              # Documentation
│   ├── 📄 paper-outline.md               # This outline
│   ├── 📄 writing-guidelines.md          # IEEE formatting tips
│   ├── 📄 submission-checklist.md        # Pre-submission checklist
│   └── 📄 conference-requirements.md     # FLICS 2026 specs
│
├── 📁 supplementary/                     # Supplementary materials
│   ├── 📄 full-slr-methodology.pdf       # Extended methodology
│   ├── 📄 complete-barrier-table.pdf     # Full barrier data
│   └── 📄 prisma-checklist.pdf           # PRISMA compliance
│
└── 📁 archive/                           # Previous versions
    ├── 📁 slr-complete-v3/               # Original SLR paper
    └── 📁 extended-abstract/             # Previous abstract
```

---

## 🔧 Tool Configuration

### 1. GitVersion Configuration (`.gitversion.yml`)

```yaml
mode: ContinuousDeployment
branches:
  main:
    regex: ^main$
    mode: ContinuousDeployment
    tag: ''
    increment: Minor
  feature:
    regex: ^feature/
    mode: ContinuousDeployment
    tag: alpha
    increment: Minor
  develop:
    regex: ^develop$
    mode: ContinuousDeployment
    tag: beta
    increment: Minor
commit-message-incrementing: Enabled
major-version-bump-message: '\+semver:\s?(major|breaking)'
minor-version-bump-message: '\+semver:\s?(minor|feature)'
patch-version-bump-message: '\+semver:\s?(patch|fix)'
```

### 2. VS Code Workspace Settings (`.vscode/settings.json`)

```json
{
  "files.associations": {
    "*.tex": "latex"
  },
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    }
  ],
  "editor.wordWrap": "on",
  "editor.rulers": [80, 100],
  "markdown.preview.breaks": true,
  "git.enableSmartCommit": true,
  "git.confirmSync": false
}
```

### 3. Git Ignore (`.gitignore`)

```gitignore
# LaTeX
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
*.fdb_latexmk
*.fls

# OS
.DS_Store
Thumbs.db

# IDE
.idea/
*.swp
*.swo

# Build
build/
dist/

# Temporary
*.tmp
*.bak
```

---

## 🔄 Workflow: VS Code ↔ Overleaf ↔ GitHub

### Option A: Overleaf-GitHub Sync (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│    VS Code (Local)                                              │
│    ├── Edit .md files, figures, data                           │
│    ├── Git commit & push to GitHub                             │
│    └── Review/merge PRs                                        │
│              │                                                  │
│              ▼                                                  │
│    GitHub Repository                                            │
│    ├── Central source of truth                                 │
│    ├── GitVersion tagging                                      │
│    └── Branch protection (main)                                │
│              │                                                  │
│              ▼ (Overleaf GitHub Sync)                          │
│    Overleaf                                                     │
│    ├── Edit .tex files collaboratively                         │
│    ├── Real-time preview                                       │
│    └── Push changes back to GitHub                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Setup Steps:

1. **Create GitHub repo**: `FL-EHDS-FLICS2026`
2. **Enable Overleaf sync**:
   - Overleaf → New Project → Import from GitHub
   - Select repository
   - Configure sync direction
3. **Local VS Code setup**:
   - Clone repo: `git clone https://github.com/[user]/FL-EHDS-FLICS2026.git`
   - Install extensions: LaTeX Workshop, GitLens, Markdown All in One
4. **Configure GitVersion**:
   - Install: `dotnet tool install --global GitVersion.Tool`
   - Run: `gitversion` to verify

---

## 📋 Development Phases

### Phase 1: Setup (Day 1-2)
- [ ] Create GitHub repository
- [ ] Setup Overleaf project with GitHub sync
- [ ] Configure GitVersion
- [ ] Import existing materials from SLR

### Phase 2: Framework Development (Day 3-7)
- [ ] Finalize FL-EHDS architecture diagram
- [ ] Write framework.tex (Section 3)
- [ ] Create Figure 2 (main contribution)
- [ ] Document component specifications

### Phase 3: Evidence Integration (Day 8-10)
- [ ] Extract key findings from SLR
- [ ] Create barrier taxonomy table
- [ ] Write evidence.tex (Section 4)
- [ ] Compress methodology description

### Phase 4: Paper Assembly (Day 11-14)
- [ ] Write introduction.tex
- [ ] Write background.tex
- [ ] Write roadmap.tex
- [ ] Write discussion.tex
- [ ] Compile and check page count

### Phase 5: Polish & Submit (Day 15-19)
- [ ] Internal review
- [ ] Figure refinement
- [ ] Reference formatting (IEEE style)
- [ ] Final page count verification (≤8)
- [ ] EasyChair submission

---

## 📅 Timeline to Deadline

| Date | Milestone | GitVersion Tag |
|------|-----------|----------------|
| Feb 1 | Project setup complete | v0.1.0 |
| Feb 5 | Framework section draft | v0.2.0 |
| Feb 8 | Evidence section draft | v0.3.0 |
| Feb 12 | Full paper draft | v0.4.0 |
| Feb 15 | Internal review complete | v0.5.0 |
| Feb 18 | Final revisions | v0.9.0 |
| Feb 20 | **Submission** | v1.0.0 |

---

## 📝 Commit Message Convention

```
<type>(<scope>): <subject>

Types:
- feat: New feature/content
- fix: Bug fix/correction
- docs: Documentation
- style: Formatting
- refactor: Restructuring
- fig: Figure changes

Examples:
- feat(framework): Add FL orchestration layer description
- fix(evidence): Correct barrier prevalence percentages
- fig(arch): Update main architecture diagram
- docs(readme): Add development workflow
```

---

## 🏷️ Branch Strategy

```
main                    # Production-ready versions
├── develop             # Integration branch
│   ├── feature/framework-layer1
│   ├── feature/framework-layer2
│   ├── feature/evidence-synthesis
│   └── feature/roadmap
└── release/v1.0        # Pre-submission freeze
```

---

## ✅ Pre-Submission Checklist

### Content
- [ ] Abstract ≤ 150 words
- [ ] Paper ≤ 8 pages (including refs)
- [ ] All figures readable at column width
- [ ] All tables fit within margins
- [ ] References in IEEE format

### Technical
- [ ] PDF compiles without errors
- [ ] All figures embedded (not linked)
- [ ] Fonts embedded
- [ ] No overfull/underfull warnings

### Compliance
- [ ] Author information complete
- [ ] ORCID included
- [ ] Acknowledgments section
- [ ] No identifying information in blind review (if applicable)

### Submission
- [ ] EasyChair account created
- [ ] Track selected (Main Track 1 or FLHA Workshop)
- [ ] Keywords entered
- [ ] PDF uploaded
- [ ] Confirmation email received

