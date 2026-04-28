# Unified AI Metallurgy System — Living Project Notes
> Updated progressively. Every architectural decision, requirement, and constraint captured here.

---

## Owner Profile
- **Status:** PhD student, solo researcher
- **Budget:** $0 (zero)
- **Hours/day:** 10+ (full sprint)
- **Goal:** Complete project → submit paper → attract supervisor
- **Timeline:** ~12 weeks

---

## ⚡ MODULE 3 — PHYSICS FILTER: CRITICAL REQUIREMENTS

### Decision logged: [Session 1]

**User requirement (verbatim intent):**
> "I want the physics module to test EVERY feature. If required, search formulas from the web at that instant to scrape data and formula. Like if the user needs a specific property like semiconducting or other features, it should understand and apply ALL types of science to extract data."

---

### What This Means Architecturally

Module 3 is NOT a static rule-checker. It is a **dynamic, self-extending physics reasoning engine**.

#### Core Design Principle:
- At runtime, Module 3 receives a property target (e.g., `semiconducting`, `piezoelectric`, `biocompatible`, `superconducting`)
- It must IDENTIFY which branch(es) of science apply to that property
- It must RETRIEVE the relevant formulas/criteria (from internal knowledge OR live web scrape)
- It must APPLY those formulas to the candidate alloy
- It must RETURN a pass/fail + confidence score with the formula/source cited

---

### Science Domains Module 3 Must Cover

#### 1. Thermodynamics & Phase Stability
- CALPHAD phase diagram checks (pycalphad)
- Gibbs free energy minimization: ΔG = ΔH - TΔS
- Hume-Rothery rules (solid solution limits)
- Formation enthalpy from DFT surrogate (MACE/CHGNet)
- Spinodal decomposition tendency
- Melting point estimation (Lindemann criterion)
- Mixing entropy for high-entropy alloys

#### 2. Quantum Mechanics / Electronic Structure
- Band gap prediction (for semiconducting property requests)
- Fermi level position
- Density of states (DOS) at Fermi level
- Effective mass tensor
- Charge carrier mobility estimation
- Work function
- Electron affinity

#### 3. Solid State Physics
- Crystal structure determination (via pymatgen structure predictor)
- Space group and symmetry
- Lattice parameter estimation
- Phonon stability (imaginary modes = unstable structure)
- Debye temperature
- Bulk modulus, shear modulus (from elastic constants)
- Poisson's ratio
- Thermal conductivity estimation

#### 4. Electrochemistry / Corrosion Science
- Pourbaix diagram stability (pymatgen has this built in)
- Standard electrode potential
- Galvanic series position
- Pitting resistance equivalent number (PREN = %Cr + 3.3×%Mo + 16×%N)
- Passive film stability
- Corrosion rate (Stern-Geary equation)

#### 5. Magnetism
- Curie temperature estimation
- Magnetic moment per atom (Hund's rules)
- Exchange interaction (Heisenberg model)
- Saturation magnetization
- Coercivity classification (soft vs hard magnetic)

#### 6. Superconductivity
- BCS critical temperature estimation (McMillan formula)
- Electron-phonon coupling constant λ
- Cooper pair formation conditions
- Type I vs Type II classification

#### 7. Piezoelectric / Ferroelectric
- Point group symmetry check (non-centrosymmetric required)
- Piezoelectric tensor components
- Spontaneous polarization
- Curie temperature (ferroelectric)

#### 8. Optical Properties
- Refractive index
- Absorption coefficient
- Optical band gap (Tauc plot equivalent)
- Reflectivity
- Transparency window

#### 9. Mechanical / Metallurgical
- Taylor factor (texture-dependent yield)
- Hall-Petch relationship: σ_y = σ_0 + k·d^(-1/2)
- Ashby map positioning
- Stacking fault energy
- Peierls stress
- Fracture toughness KIC
- Fatigue strength ratio

#### 10. Biocompatibility (for biomedical alloys)
- Element toxicity classification (WHO limits)
- Ion release rate prediction
- Cytotoxicity index
- Osseointegration potential score

---

### The Dynamic Formula Retrieval System

When a property is requested that Module 3 doesn't have hardcoded:

```
STEP 1: Classify the property → which science domain?
         (LLM-based classifier, few-shot prompted)

STEP 2: Check internal formula registry
         → if formula found: apply it

STEP 3: If NOT found in registry:
         → Web scrape: search "{property} formula materials science"
         → Sources: Wikipedia, NIST WebBook, Springer Materials snippets, arXiv
         → Parse formula using SymPy (symbolic math)
         → Add to formula registry for future use

STEP 4: Extract required input parameters from alloy composition
         → Some from pymatgen, some from matminer features, some from ML prediction

STEP 5: Apply formula → compute property value

STEP 6: Compare to user threshold → PASS / FAIL + confidence

STEP 7: Return: {property, value, formula_used, source_url, pass/fail, confidence}
```

---

### Key Libraries for This Module

| Library | Role |
|---|---|
| `pycalphad` | Thermodynamics, phase diagrams |
| `pymatgen` | Crystal structure, Pourbaix diagrams, band structure |
| `matminer` | 145 composition features (electronegativity, atomic radius, etc.) |
| `smact` | Charge neutrality, oxidation state checks |
| `SymPy` | Symbolic math — parse and evaluate scraped formulas |
| `BeautifulSoup` / `httpx` | Web scraping for dynamic formula retrieval |
| `MACE` or `CHGNet` | Universal interatomic potential — free DFT surrogate |
| `aflow-ml` | Property prediction from AFLOW ML models |
| LLaMA 3 / Mistral (via Ollama) | Property → science domain classification |

---

### Formula Registry Structure (JSON)

```json
{
  "pitting_resistance": {
    "formula": "Cr + 3.3*Mo + 16*N",
    "variables": {"Cr": "wt%", "Mo": "wt%", "N": "wt%"},
    "threshold": ">25 for marine, >40 for aggressive",
    "source": "ASTM G48, Outokumpu Corrosion Handbook",
    "domain": "electrochemistry"
  },
  "hall_petch": {
    "formula": "sigma_y = sigma_0 + k * d**(-0.5)",
    "variables": {"sigma_0": "friction stress MPa", "k": "Hall-Petch constant", "d": "grain size m"},
    "source": "Hall (1951), Petch (1953)",
    "domain": "mechanical"
  }
}
```

---

### Property → Science Domain Mapping (Router Table)

| User Says | Domains Activated |
|---|---|
| "semiconducting" | Electronic structure, QM, optical |
| "corrosion resistant" | Electrochemistry, thermodynamics |
| "biocompatible" | Biocompatibility, corrosion, toxicology |
| "magnetic" | Magnetism, electronic structure |
| "superconducting" | Superconductivity, QM, phonons |
| "lightweight high strength" | Mechanical, thermodynamics |
| "high temperature" | Thermodynamics, creep, phase stability |
| "piezoelectric" | Piezo/ferroelectric, symmetry, QM |
| "transparent" | Optical, electronic structure |
| "wear resistant" | Mechanical, tribology |
| "shape memory" | Phase transformation, martensitic, thermodynamics |

---

## Other Modules — Notes (TBD, will expand)

### Module 1 (LLM Interpreter)
- Use Mistral-7B via Ollama (local, free)
- Output: structured JSON property target
- TBD: further requirements

### Module 2 (Generative Designer)
- VAE first, upgrade to diffusion if validity <70%
- TBD: further requirements

### Module 4 (Property Predictor)
- CGCNN baseline → multi-task → ensemble UQ
- TBD: further requirements

### Module 5 (Optimization Engine)
- BoTorch + NSGA-III
- TBD: further requirements

---

## Open Questions (to be resolved during build)
- [ ] Which CALPHAD open database covers the widest alloy systems in pycalphad?
- [ ] How to handle properties where no formula exists (e.g., novel metamaterial behavior)?
- [ ] Should formula registry be version-controlled separately?
- [ ] What is the performance budget per candidate for Module 3? (Target: <5s per alloy)

---

*Last updated: Session 1 — Physics Module requirements captured*
