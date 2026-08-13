# 🎓 Nigerian University Payroll Category Predictor

A complete multi-class ML project predicting the payroll category of Nigerian university academic staff across four bands — **Low, Medium, High and Executive** — from staff profile information alone.

## 🌐 Live Demo
**[Try the app →](https://payrollproject-z1qq.onrender.com)**

---

## 📌 Project Overview
Nigerian university salary administration under the NEEDS (National Universities Commission Earned Academic Allowances) structure ties compensation directly to academic rank and experience. This system enables HR departments to classify staff payroll bands automatically from profile data — without requiring access to sensitive salary figures.

**What makes this project stand out:**
- Target column had **61 different formats** across just 4 classes — most complex cleaning challenge yet
- Salary columns (BaseSalary, NetSalary, GrossSalary) deliberately **excluded** from model features to prevent near-leakage
- **Domain-aware cleaning** — CoursesTaught clipped at upper bound only (0 is valid for research professors)
- **4 engineered features** — Performance composite ranked 4th overall, outperforming all categorical columns

---

## 📊 Dataset
| Property | Value |
|---|---|
| Rows | 20,250 (after duplicate removal) |
| Columns | 26 |
| Target | PayrollCategory: Low / Medium / High / Executive |
| Key Challenge | 61 formats in target column alone |

---

## 🧹 Data Cleaning Highlights
| Column | Problem | Solution |
|---|---|---|
| BaseSalary / NetSalary / GrossSalary | NGN, ₦, #, naira formats; 'k' suffix; ×100 errors | Strip symbols; k→×1000; IQR clip |
| Age | 'yrs','years' suffixes; impossible values | Strip suffixes; clip impossible values to NaN |
| YearsOfExperience | 'approx','fresh','new staff' formats | Strip keywords; fresh→0; group mean by Rank |
| TaxDeduction | Some stored as '12.5%' not naira | Detect %; ÷100; × gross salary |
| Gender | 14+ formats: 'Man','M','masculine','1' | Map to Male/Female; unknown→NaN |
| AbsenceDays | Negatives; 'nil'; 200+ days | np.abs(); nil→0; clip 0-261 |
| CoursesTaught | 'N/A','courses' suffix; 0-50 range | Strip; clip UPPER only — 0 is valid! |
| PayrollCategory | **61 different formats** | Complete function-based dictionary mapping |
| Duplicates | 200 hidden rows | drop_duplicates() at start |

### Key Cleaning Insight
```python
# CoursesTaught — clip upper bound ONLY
# 0 is valid for research-focused professors
upper = df["CoursesTaught"].quantile(0.75) + 1.5 * IQR
df["CoursesTaught"] = df["CoursesTaught"].clip(0, upper)
```

---

## ⚙️ Feature Engineering
| Feature | Formula | Importance Rank |
|---|---|---|
| Performance | (YearsExp + Publications) / (1 + PerformanceScore) | 4th — 10.87% |
| ResearchProductivity | Publications / (1 + YearsOfExperience) | 7th — 2.35% |
| TeachingLoad | CoursesTaught / (1 + StudentsSupervised) | 9th — 2.17% |
| YearsStudents | StudentsSupervised / (1 + YearsOfExperience) | 8th — 2.25% |

**Performance (engineered) outranked ALL categorical columns** — including Department, University and State.

---

## 🤖 Model Results
| Model | Accuracy | Low F1 | Notes |
|---|---|---|---|
| Logistic Regression | 86.68% | 0.68 | Only 0.88% behind RF — linear data! |
| Decision Tree | 85.00% | 0.63 | Weakest — single tree limitation |
| **Random Forest** | **87.56%** | **0.71** | **BEST — Deployed ✅** |

### Why LR competed so closely with RF:
Academic rank in Nigerian universities maps almost directly to salary band under the NEEDS structure — strong linear relationships mean LR is naturally competitive.

### Why Low class had lowest F1 (0.68-0.71):
Low sits at the salary boundary where staff characteristics overlap most with Medium and Executive bands — a classic boundary class problem.

---

## 📊 Feature Importance (Random Forest)
```
Rank                 → 41.49%  ← Dominates (NEEDS structure)
YearsOfExperience    → 16.15%
Age                  → 11.77%
Performance          → 10.87%  ← Engineered — ranked 4th!
StudentsSupervised   →  5.21%
Publications         →  3.66%
ResearchProductivity →  2.35%  ← Engineered
YearsStudents        →  2.25%  ← Engineered
TeachingLoad         →  2.17%  ← Engineered
```

---

## 🏗️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn (Random Forest, Logistic Regression, Decision Tree)
- **Web Backend:** Flask
- **Frontend:** HTML5, CSS3 (Dark Academic Blue Theme)
- **Deployment:** Render.com
- **Version Control:** GitHub

---

## 📁 Project Structure
```
PayrollPredictor/
├── data/
│   └── nigerian_university_payroll_messy.csv
├── models/
│   ├── RandomForeModel.joblib
│   └── Preprocessor1.joblib
├── templates/
│   └── payroll.html
├── static/
│   └── payroll_style.css
├── app.py
├── requirements.txt
└── Procfile
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/DavidGabriel213/PayrollPredictor
cd PayrollPredictor
pip install -r requirements.txt
python app.py
```

---

## 💡 Key Learnings
1. **61 target formats** — new personal record, surpassing 28 formats in Student Dropout project
2. **Domain-aware clipping** — zero CoursesTaught is valid; never clip blindly without domain knowledge
3. **Exclude derived columns** — NetSalary excluded because target was computed from it
4. **Rank dominance (41.5%)** — strongest single-feature importance across all 8 projects built
5. **Engineered features beat raw** — Performance composite outranked all categorical columns
6. **LR vs RF gap: 0.88%** — narrow gap confirms linear relationships in Nigerian salary data
7. **Built after 2 months away from coding** — returned without referencing external resources

---

## 👨‍💻 About
**Gabriel David** | Mathematics Undergraduate | ATBU Bauchi
Self-taught ML Engineer — 8th deployed project. Built on Android phone.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-gabriel--david--ds-blue)](https://linkedin.com/in/gabriel-david-ds)
[![GitHub](https://img.shields.io/badge/GitHub-DavidGabriel213-black)](https://github.com/DavidGabriel213)

