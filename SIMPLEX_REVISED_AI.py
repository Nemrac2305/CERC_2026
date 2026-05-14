import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import json
import requests
import threading

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ===================== Localizare interfata EN / RO =====================
UI_TRANSLATIONS = {
    "Revised Simplex Pro V9 - Azure Edition": "Simplex Revizuit Pro V9 - Ediție Azure",
    "1. Setup & Input": "1. Configurare & Date",
    "2. Iteration Process": "2. Proces iterativ",
    "3. Results & Analysis": "3. Rezultate & Analiză",
    "4. Verification & Charts": "4. Verificare & Grafice",
    "Problem setup": "Configurarea problemei",
    "Objective type:": "Tip obiectiv:",
    "Number of variables (n):": "Număr de variabile (n):",
    "Number of constraints (m):": "Număr de restricții (m):",
    "Load JSON": "Încarcă JSON",
    "Save JSON": "Salvează JSON",
    "Generate table": "Generează tabel",
    "Load example": "Încarcă exemplu",
    "Next step": "Pasul următor",
    "Solve completely": "Rezolvă complet",
    "Reset run": "Resetează rularea",
    "AI Assistant": "Asistent AI",
    "Problem matrix": "Matricea problemei",
    "Help": "Ajutor",
    "Current state preview": "Previzualizarea stării curente",
    "Technical log": "Jurnal tehnic",
    "Current basis B": "Baza curentă B",
    "Lower triangular factor L": "Factor triunghiular inferior L",
    "Upper triangular factor U": "Factor triunghiular superior U",
    "Standardized matrix A": "Matricea standardizată A",
    "Simplex multiplier π": "Multiplicator simplex π",
    "Reduced costs Δ = c - z": "Costuri reduse Δ = c - z",
    "Direction d = B⁻¹P_k / Ratios θ_i": "Direcția d = B⁻¹P_k / Rapoarte θ_i",
    "Iteration summary": "Rezumatul iterației",
    "Final value f(x*):": "Valoarea finală f(x*):",
    "Solution vector x*": "Vectorul soluție x*",
    "Constraint check Ax* ? b": "Verificare restricții Ax* ? b",
    "Mathematical verification": "Verificare matematică",
    "Verification A · x vs b": "Verificare A · x vs b",
    "Verification A_ext · x_ext vs b": "Verificare A_ext · x_ext vs b",
    "Z convergence chart by iteration": "Graficul convergenței Z pe iterații",
    "Monotone convergence of Z_ext = c_B^T x_B": "Convergența monotonă a Z_ext = c_B^T x_B",
    "Iteration": "Iterația",
    "Z_ext = c_B^T x_B": "Z_ext = c_B^T x_B",
    "Variable": "Variabilă",
    "Type": "Tip",
    "Value": "Valoare",
    "Constraint": "Restricție",
    "LHS": "Partea stângă",
    "Sign": "Semn",
    "RHS": "Partea dreaptă",
    "Status": "Stare",
    "Input for the revised simplex method": "Date pentru metoda simplex revizuită",
    "CERC notation: x for variables, y for slack/surplus variables, z for artificial variables; keyboard navigation and JSON persistence.": "Notație CERC: x pentru variabile, y pentru variabile de ecart/surplus, z pentru variabile artificiale; navigare cu tastatura și persistență JSON.",
    "Objective function f(x) =": "Funcția obiectiv f(x) =",
    "System Ax = b after standardization": "Sistemul Ax = b după standardizare",
    "Row i": "Rând i",
    "Note: standardization introduces y variables for slack/surplus and z variables for artificial variables penalized by M.": "Notă: standardizarea introduce variabile y pentru ecart/surplus și variabile z pentru artificiale penalizate cu M.",
    "No iteration has been run yet.": "Nu a fost rulată nicio iterație.",
    "No iteration data available yet.": "Nu există încă date de iterație.",
    "No completed solve yet.": "Nu există încă o rezolvare completă.",
    "No verification available.": "Nu există încă verificare.",
    "Last state: iteration": "Ultima stare: iterația",
    "Iteration:": "Iterația:",
    "Status:": "Stare:",
    "Dynamic M:": "M dinamic:",
    "Basis B:": "Baza B:",
    "Entering variable x_k:": "Variabila care intră x_k:",
    "Leaving variable from B:": "Variabila care iese din B:",
    "The optimality condition Δ_j ≤ 0 is satisfied for all nonbasic variables.": "Condiția de optimalitate Δ_j ≤ 0 este satisfăcută pentru toate variabilele nebazice.",
    "Entering variable x_k =": "Variabila care intră x_k =",
    "Leaving variable from B:": "Variabila care iese din B:",
    "Charnes lexicographic rule applied": "Regula lexicografică Charnes aplicată",
    "Partial f(x*) =": "f(x*) parțial =",
    "Convention: Δ_j = c_j - z_j, with z_j = π^T P_j": "Convenție: Δ_j = c_j - z_j, cu z_j = π^T P_j",
    "Solution found:": "Soluție găsită:",
    "The constraint system is infeasible.": "Sistemul de restricții este infezabil.",
    "The problem is unbounded in the direction of the entering variable.": "Problema este nemărginită în direcția variabilei care intră.",
    "Original system feasibility:": "Fezabilitatea sistemului original:",
    "Verification A_ext · x_ext = b:": "Verificare A_ext · x_ext = b:",
    "CERC notation:": "Notație CERC:",
    "Notation:": "Notație:",
    "Decision variables": "Variabile de decizie",
    "Active constraints": "Restricții active",
    "Nonbinding constraints": "Restricții neactive",
    "AI Optimization Assistant": "Asistent AI pentru optimizare",
    "Send": "Trimite",
    "Import JSON": "Importă JSON",
    "Clear Chat": "Șterge chatul",
    "Explain solution": "Explică soluția",
    "Explain your optimization problem in natural language.\nExample: A company produces phones and laptops with labor and material constraints.\nThe AI will generate a JSON problem directly compatible with the simplex solver.": "Descrie problema de optimizare în limbaj natural.\nExemplu: O companie produce telefoane și laptopuri cu restricții de muncă și materiale.\nAI-ul va genera o problemă JSON compatibilă direct cu solverul simplex.",
    "USER:": "UTILIZATOR:",
    "AI ERROR:": "EROARE AI:",
    "No JSON": "Niciun JSON",
    "Import successful": "Import reușit",
    "Import failed": "Import eșuat",
    "No solution": "Nicio soluție",
    "AI is preparing a simple explanation of the current solution...": "AI pregătește o explicație simplă a soluției curente...",
    "AI generated and validated this JSON problem. You can import it now.": "AI a generat și a validat această problemă JSON. O poți importa acum.",
    "Status: optimal": "Stare: optim",
    "Status: infeasible": "Stare: infezabil",
    "Status: unbounded": "Stare: nemărginit",
    "Status: continue": "Stare: continuă",
    "x (decision)": "x (decizie)",
    "y (slack)": "y (ecart)",
    "y (surplus)": "y (surplus)",
    "z (artificial)": "z (artificială)",
    "binding": "activă",
    "nonbinding": "neactivă",
    "violated": "încălcată",
    "optimal": "optim",
    "infeasible": "infezabil",
    "unbounded": "nemărginit",
}


# Traduceri suplimentare pentru continut din Text/ScrolledText si mesaje dinamice.
# Acestea sunt aplicate si peste jurnalul deja afisat atunci cand se schimba limba.
UI_TRANSLATIONS.update({
    "• Use <=, >=, or = for each constraint.\n• Navigation: arrow keys + Enter between cells.\n• If b is negative, the solver automatically normalizes the row.\n• Artificial variables are handled with the Big-M method.":
        "• Folosește <=, >= sau = pentru fiecare restricție.\n• Navigare: tastele săgeată + Enter între celule.\n• Dacă b este negativ, solverul normalizează automat rândul.\n• Variabilele artificiale sunt tratate prin metoda Big-M.",
    "No leaving variable exists, so the problem is unbounded in this direction.":
        "Nu există o variabilă care să iasă, deci problema este nemărginită în această direcție.",
    "No leaving variable | unbounded direction": "Nicio variabilă nu iese | direcție nemărginită",
    "none (unbounded direction)": "niciuna (direcție nemărginită)",
    "f(x) real:": "f(x) reală:",
    "c_B =": "c_B =",
    "x_B =": "x_B =",
    "Z_ext = c_B^T x_B:": "Z_ext = c_B^T x_B:",
    "No leaving variable": "Nicio variabilă nu iese",
    "unbounded direction": "direcție nemărginită",
    "The revised simplex generator was initialized according to the CERC notation.":
        "Generatorul simplex revizuit a fost inițializat conform notației CERC.",
    "Active solver: UniversalSolverLU (dynamic Big M + manual LU + Charnes rule).":
        "Solver activ: UniversalSolverLU (Big-M dinamic + LU manual + regula Charnes).",
    "Problem saved to:": "Problema a fost salvată în:",
    "Problem loaded from:": "Problema a fost încărcată din:",
    "Example loaded:": "Exemplu încărcat:",
    "Expected result:": "Rezultat așteptat:",
    "--- Iteration": "--- Iterația",
    "The dynamic numerical barrier was set to M =": "Bariera numerică dinamică a fost setată la M =",
    "Current basis:": "Baza curentă:",
    "PLU factorization computed for P * B = L * U.": "Factorizarea PLU a fost calculată pentru P * B = L * U.",
    "Original objective value (without penalties) =": "Valoarea funcției obiectiv originale (fără penalizări) =",
    "Z_ext (with M penalties, used for the chart) =": "Z_ext (cu penalizări M, folosit pentru grafic) =",
    "Degeneracy detected. Applying the Charnes tie-breaking rule.":
        "A fost detectată degenerare. Se aplică regula Charnes pentru departajare.",
    "Leaving candidates:": "Candidați pentru ieșire:",
    "Entering ": "Intră ",
    "Leaving ": "Iese ",
    "theta =": "theta =",
    "Iteration status: continue": "Starea iterației: continuă",
    "Iteration status: optimal": "Starea iterației: optim",
    "Iteration status: infeasible": "Starea iterației: infezabil",
    "Iteration status: unbounded": "Starea iterației: nemărginit",
    "Iteration status:": "Starea iterației:",
    "Final result:": "Rezultat final:",
    "ERROR:": "EROARE:",
    "Iteration ": "Iterația ",
    "Status: ": "Stare: ",
    "Partial f(x*) =": "f(x*) parțial =",
    "Basis B:": "Baza B:",
    "with z_j =": "cu z_j =",
    "No iteration has been run yet.": "Nu a fost rulată nicio iterație.",
    "No iteration data available yet.": "Nu există încă date de iterație.",
    "No completed solve yet.": "Nu există încă o rezolvare completă.",
    "No verification available.": "Nu există încă verificare.",
})

# Mesaje suplimentare folosite de fereastra AI si dialogurile ei.
UI_TRANSLATIONS.update({
    "The AI has not generated a JSON problem yet.":
        "AI-ul nu a generat încă o problemă JSON.",
    "The generated JSON could not be imported.":
        "JSON-ul generat nu a putut fi importat.",
    "The AI-generated optimization problem was loaded.":
        "Problema de optimizare generată de AI a fost încărcată.",
    "Optimization problem imported from AI Assistant.":
        "Problema de optimizare a fost importată din Asistentul AI.",
    "Solve the problem first, then AI can explain the final mathematical solution.":
        "Rezolvă mai întâi problema, apoi AI-ul poate explica soluția matematică finală.",
    "Could not generate a complete JSON problem.":
        "Nu s-a putut genera o problemă JSON completă.",
    "Make sure Ollama is installed and running.":
        "Asigură-te că Ollama este instalat și rulează.",
    "Run in terminal:":
        "Rulează în terminal:",
    "Technical details:":
        "Detalii tehnice:",
    "The AI did not return a complete valid JSON problem. Last error:":
        "AI-ul nu a returnat o problemă JSON completă și validă. Ultima eroare:",
    "Empty explanation.":
        "Explicație goală.",
})


RO_TO_EN_TRANSLATIONS = {value: key for key, value in UI_TRANSLATIONS.items()}


def translate_text_for_language(text, language):
    """Translate known interface fragments without changing user-entered data."""
    if not isinstance(text, str) or not text:
        return text
    mapping = UI_TRANSLATIONS if language == "ro" else RO_TO_EN_TRANSLATIONS
    if text in mapping:
        return mapping[text]
    translated = text
    exact_only = {
        "optimal", "infeasible", "unbounded", "continue", "binding", "nonbinding", "violated",
        "optim", "infezabil", "nemărginit", "continuă", "activă", "neactivă", "încălcată",
    }
    for source, target in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        if source in exact_only:
            continue
        if source and source in translated:
            translated = translated.replace(source, target)
    return translated


class UniversalSolverLU:
    """
    Motor matematic pentru Simplex Revizuit Universal.

    Caracteristici:
    - suporta max / min;
    - suporta restrictii de tip <=, >=, =;
    - foloseste metoda M-Mare pentru variabile artificiale;
    - foloseste factorizare LU manuala pentru rezolvarea sistemelor liniare;
    - expune algoritmul ca generator, astfel incat interfata sa poata rula pas cu pas.
    """

    def __init__(self, objective="max", tol=1e-10, max_iter=100, big_m=1e6):
        self.objective = objective.lower().strip()
        if self.objective not in {"max", "min"}:
            raise ValueError("Objective type must be 'max' or 'min'.")
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.M = float(big_m)

    # ===================== Algebra liniara manuala (LU) =====================
    def plu_decomposition(self, matrix):
        """Partial-pivoting LU factorization: P * B = L * U."""
        A = np.array(matrix, dtype=float, copy=True)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("The basis matrix must be square for PLU factorization.")

        n = A.shape[0]
        P = np.eye(n, dtype=float)
        L = np.eye(n, dtype=float)
        U = A.copy()

        for k in range(n):
            pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
            pivot_value = U[pivot_row, k]
            if abs(pivot_value) < self.tol:
                raise ValueError(
                    "The current basis is singular or nearly singular even with partial pivoting. "
                    "The model may contain linearly dependent constraints."
                )

            if pivot_row != k:
                U[[k, pivot_row], :] = U[[pivot_row, k], :]
                P[[k, pivot_row], :] = P[[pivot_row, k], :]
                if k > 0:
                    L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

            for i in range(k + 1, n):
                L[i, k] = U[i, k] / U[k, k]
                U[i, k:] = U[i, k:] - L[i, k] * U[k, k:]
                U[i, k] = 0.0

        return P, L, U

    def solve_plu(self, P, L, U, b):
        """Solve B * x = b given P * B = L * U."""
        b = np.array(b, dtype=float)
        rhs = P @ b
        n = len(rhs)

        y = np.zeros(n, dtype=float)
        for i in range(n):
            y[i] = rhs[i] - np.dot(L[i, :i], y[:i])

        x = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            if abs(U[i, i]) < self.tol:
                raise ValueError("Near-zero pivot encountered during backward substitution.")
            x[i] = (y[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]
        return x

    def solve_transposed_plu(self, P, L, U, b):
        """Solve B^T * x = b given P * B = L * U."""
        b = np.array(b, dtype=float)
        n = len(b)

        y = np.zeros(n, dtype=float)
        for i in range(n):
            if abs(U[i, i]) < self.tol:
                raise ValueError("Near-zero pivot encountered in the transposed system.")
            y[i] = (b[i] - np.dot(U[:i, i], y[:i])) / U[i, i]

        w = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            w[i] = y[i] - np.dot(L[i + 1 :, i], w[i + 1 :])

        return P.T @ w

    # ===================== Standardizare cu M-Mare =====================
    def _normalize_rows(self, A, b, signs):
        """
        Daca o restrictie are b < 0, inmultim toata linia cu -1 si inversam semnul.
        Astfel, baza initiala construita din slack/artificiale ramane coerenta.
        """
        A_norm = np.array(A, dtype=float, copy=True)
        b_norm = np.array(b, dtype=float, copy=True)
        signs_norm = list(signs)

        for i in range(len(b_norm)):
            if b_norm[i] < -self.tol:
                A_norm[i, :] *= -1.0
                b_norm[i] *= -1.0
                if signs_norm[i] == "<=":
                    signs_norm[i] = ">="
                elif signs_norm[i] == ">=":
                    signs_norm[i] = "<="

        return A_norm, b_norm, signs_norm

    def standardize(self, c, A, b, signs):
        """
        Transforma problema in forma standard prin adaugarea de:
        - variabile de ecart pentru <=,
        - variabile de surplus + artificiale pentru >=,
        - variabile artificiale pentru =.
        """
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)
        signs = list(signs)

        if A.ndim != 2 or b.ndim != 1 or c.ndim != 1:
            raise ValueError("A must be a matrix, while b and c must be vectors.")
        if A.shape[0] != len(b):
            raise ValueError("The number of rows in A must match the size of b.")
        if A.shape[1] != len(c):
            raise ValueError("The number of columns in A must match the size of c.")
        if len(signs) != len(b):
            raise ValueError("A sign must be specified for each constraint.")
        if any(s not in {"<=", ">=", "="} for s in signs):
            raise ValueError("The allowed signs are only: <=, >=, =.")

        A, b, signs = self._normalize_rows(A, b, signs)
        m, n_orig = A.shape

        # Big M dinamic: suficient de mare fata de scara coeficientilor obiectivului,
        # dar nu absurd de mare pentru a agrava erorile numerice.
        self.M = float((np.sum(np.abs(c)) + 1.0) * 1000.0)

        A_ext = np.array(A, dtype=float, copy=True)
        c_ext = np.array(c, dtype=float, copy=True)
        basis = []
        artificial_indices = []
        var_names = [f"x{j + 1}" for j in range(n_orig)]
        var_kinds = ["x (decision)" for _ in range(n_orig)]
        row_var_info = []

        m_cost = -self.M if self.objective == "max" else self.M

        for i in range(m):
            sign = signs[i]
            info = {"row": i, "sign": sign, "slack": None, "surplus": None, "artificial": None}

            if sign == "<=":
                col = np.zeros((m, 1), dtype=float)
                col[i, 0] = 1.0
                A_ext = np.hstack([A_ext, col])
                c_ext = np.append(c_ext, 0.0)
                idx = A_ext.shape[1] - 1
                basis.append(idx)
                var_names.append(f"y{i + 1}")
                var_kinds.append("y (slack)")
                info["slack"] = idx

            elif sign == ">=":
                s_col = np.zeros((m, 1), dtype=float)
                s_col[i, 0] = -1.0
                A_ext = np.hstack([A_ext, s_col])
                c_ext = np.append(c_ext, 0.0)
                s_idx = A_ext.shape[1] - 1
                var_names.append(f"y{i + 1}⁻")
                var_kinds.append("y (surplus)")
                info["surplus"] = s_idx

                a_col = np.zeros((m, 1), dtype=float)
                a_col[i, 0] = 1.0
                A_ext = np.hstack([A_ext, a_col])
                c_ext = np.append(c_ext, m_cost)
                a_idx = A_ext.shape[1] - 1
                basis.append(a_idx)
                artificial_indices.append(a_idx)
                var_names.append(f"z{i + 1}")
                var_kinds.append("z (artificial)")
                info["artificial"] = a_idx

            else:  # sign == "="
                a_col = np.zeros((m, 1), dtype=float)
                a_col[i, 0] = 1.0
                A_ext = np.hstack([A_ext, a_col])
                c_ext = np.append(c_ext, m_cost)
                a_idx = A_ext.shape[1] - 1
                basis.append(a_idx)
                artificial_indices.append(a_idx)
                var_names.append(f"z{i + 1}")
                var_kinds.append("z (artificial)")
                info["artificial"] = a_idx

            row_var_info.append(info)

        return {
            "A": A,
            "b": b,
            "c": c,
            "signs": signs,
            "A_ext": A_ext,
            "c_ext": c_ext,
            "basis": basis,
            "artificial_indices": artificial_indices,
            "var_names": var_names,
            "var_kinds": var_kinds,
            "row_var_info": row_var_info,
            "m": m,
            "n": n_orig,
            "total_vars": A_ext.shape[1],
            "big_m": float(self.M),
            "m_cost": float(m_cost),
        }

    # ===================== Utilitare =====================
    def build_solution(self, basis, xb, total_vars):
        x = np.zeros(total_vars, dtype=float)
        for i, idx in enumerate(basis):
            x[idx] = xb[i]
        return x

    def _compute_basis_inverse(self, P, L, U):
        """Build B^-1 column by column using the current PLU factorization."""
        m = L.shape[0]
        cols = []
        for k in range(m):
            e_k = np.zeros(m, dtype=float)
            e_k[k] = 1.0
            cols.append(self.solve_plu(P, L, U, e_k))
        return np.column_stack(cols)

    def _lex_compare(self, left, right):
        """Compara doua vectori in ordinea lexicografica, cu toleranta numerica."""
        for a, b in zip(left, right):
            diff = float(a) - float(b)
            if abs(diff) <= max(self.tol, 1e-12):
                continue
            return -1 if diff < 0 else 1
        return 0

    def _select_entering_variable(self, reduced_costs, non_basis):
        """Selecteaza variabila de intrare dupa conventia Delta = c - z."""
        if self.objective == "max":
            candidates = [j for j in non_basis if reduced_costs[j] > self.tol]
            entering = max(candidates, key=lambda j: (reduced_costs[j], -j)) if candidates else None
        else:
            candidates = [j for j in non_basis if reduced_costs[j] < -self.tol]
            entering = min(candidates, key=lambda j: (reduced_costs[j], j)) if candidates else None
        return entering, candidates

    def _lexicographic_leaving_row(self, xb, B_inv, d, candidates):
        """Aplica regula lexicografica Charnes pe randurile din [x_B | B^-1] / d_i."""
        best_row = int(candidates[0])
        best_vec = np.concatenate(([xb[best_row]], B_inv[best_row, :])) / d[best_row]

        for row in candidates[1:]:
            row = int(row)
            candidate_vec = np.concatenate(([xb[row]], B_inv[row, :])) / d[row]
            if self._lex_compare(candidate_vec, best_vec) < 0:
                best_row = row
                best_vec = candidate_vec

        return best_row

    def _constraint_statuses(self, A, b, signs, x):
        lhs = A @ x
        statuses = []
        for i, sign in enumerate(signs):
            gap = lhs[i] - b[i]
            if sign == "<=":
                if abs(gap) <= 1e-7:
                    status = "binding"
                elif gap < 0:
                    status = "nonbinding"
                else:
                    status = "violated"
            elif sign == ">=":
                if abs(gap) <= 1e-7:
                    status = "binding"
                elif gap > 0:
                    status = "nonbinding"
                else:
                    status = "violated"
            else:
                status = "binding" if abs(gap) <= 1e-7 else "violated"
            statuses.append({
                "row": i,
                "lhs": float(lhs[i]),
                "rhs": float(b[i]),
                "sign": sign,
                "status": status,
            })
        return lhs, statuses

    def _build_result(self, problem, basis, xb, status, message, z_history, iterations):
        n = problem["n"]
        total_vars = problem["total_vars"]
        x_full = self.build_solution(basis, xb, total_vars)
        x_decision = x_full[:n]
        Ax, constraint_statuses = self._constraint_statuses(
            problem["A"], problem["b"], problem["signs"], x_decision
        )
        Aext_xext = problem["A_ext"] @ x_full
        standardized_ok = np.allclose(Aext_xext, problem["b"], atol=1e-7)
        feasible_original = all(item["status"] != "violated" for item in constraint_statuses)

        return {
            "status": status,
            "message": message,
            "objective": self.objective,
            "basis": basis.copy(),
            "xb": np.array(xb, dtype=float, copy=True),
            "x_full": x_full,
            "x_decision": x_decision,
            "objective_value": float(problem["c"] @ x_decision),
            "z_history": list(z_history),
            "big_m": float(problem.get("big_m", self.M)),
            "iterations": iterations,
            "A": problem["A"],
            "b": problem["b"],
            "c": problem["c"],
            "signs": problem["signs"],
            "A_ext": problem["A_ext"],
            "c_ext": problem["c_ext"],
            "Ax": Ax,
            "Aext_xext": Aext_xext,
            "feasible_original": feasible_original,
            "standardized_ok": standardized_ok,
            "constraint_statuses": constraint_statuses,
            "var_names": problem["var_names"],
            "var_kinds": problem["var_kinds"],
            "artificial_indices": problem["artificial_indices"],
            "row_var_info": problem["row_var_info"],
        }

    def solve_complete(self, c, A, b, signs):
        last_state = None
        for state in self.iteration_generator(c, A, b, signs):
            last_state = state
        if last_state is None:
            raise ValueError("The algorithm produced no iterations.")
        return last_state

    # ===================== Generatorul Simplex Revizuit =====================
    def iteration_generator(self, c, A, b, signs):
        problem = self.standardize(c, A, b, signs)
        A_ext = problem["A_ext"]
        c_ext = problem["c_ext"]
        basis = problem["basis"].copy()
        xb = np.array(problem["b"], dtype=float, copy=True)
        m = problem["m"]
        total_vars = problem["total_vars"]
        z_history = []

        for it in range(1, self.max_iter + 1):
            B = A_ext[:, basis]
            cb = c_ext[basis]

            # 1) PLU factorization of the current basis: P * B = L * U.
            P, L, U = self.plu_decomposition(B)

            # 2) The simplex multipliers satisfy B^T * pi = c_B.
            pi = self.solve_transposed_plu(P, L, U, cb)

            # 3) Pricing conform conventiei din document: Delta = c - z = c_j - pi^T a_j.
            reduced_costs = c_ext - (A_ext.T @ pi)

            # Z teoretic/extins: foloseste c_B curent, inclusiv penalizarile ±M.
            z_theoretical = float(cb @ xb)
            z_history.append(z_theoretical)

            x_full = self.build_solution(basis, xb, total_vars)
            x_decision = x_full[: problem["n"]]
            z_value = float(problem["c"] @ x_decision)

            non_basis = [j for j in range(total_vars) if j not in basis]
            entering, candidates = self._select_entering_variable(reduced_costs, non_basis)
            optimal = entering is None

            state = {
                "it": it,
                "problem": problem,
                "basis": basis.copy(),
                "xb": xb.copy(),
                "B": B.copy(),
                "cb": cb.copy(),
                "P": P.copy(),
                "L": L.copy(),
                "U": U.copy(),
                "pi": pi.copy(),
                "reduced": reduced_costs.copy(),
                "z": z_value,
                "z_ext": z_theoretical,
                "z_history": list(z_history),
                "x_full": x_full.copy(),
                "x_decision": x_decision.copy(),
                "entering": None,
                "d": None,
                "theta": None,
                "ratios": None,
                "leaving_row": None,
                "leaving_var": None,
                "status": "continue",
                "message": "Iteration computed.",
                "final_result": None,
                "lexicographic_used": False,
                "lexicographic_candidates": [],
                "delta_convention": "c-z",
            }

            if optimal:
                is_infeasible = False
                for idx in problem["artificial_indices"]:
                    if idx in basis:
                        art_val = xb[basis.index(idx)]
                        if art_val > self.tol:
                            is_infeasible = True
                            break

                state["status"] = "infeasible" if is_infeasible else "optimal"
                state["message"] = (
                    "The constraint system is infeasible."
                    if is_infeasible
                    else "The optimality condition Δ <= 0 is satisfied for the convention Δ = c - z."
                )
                state["final_result"] = self._build_result(
                    problem, basis, xb, state["status"], state["message"], z_history, it
                )
                yield state
                return

            # 4) Directia de deplasare: B * d = a_k.
            d = self.solve_plu(P, L, U, A_ext[:, entering])
            state["entering"] = entering
            state["d"] = d.copy()

            # 5) Nemarginire: daca toate componentele lui d sunt <= 0.
            if np.all(d <= self.tol):
                state["status"] = "unbounded"
                state["message"] = "The problem is unbounded in the direction of the entering variable."
                state["final_result"] = self._build_result(
                    problem, basis, xb, state["status"], state["message"], z_history, it
                )
                yield state
                return

            # 6) Testul raportului minim + regula lexicografica Charnes.
            ratios = np.array([xb[i] / d[i] if d[i] > self.tol else np.inf for i in range(m)], dtype=float)
            theta = float(np.min(ratios))
            leaving_candidates = np.where(np.isfinite(ratios) & np.isclose(ratios, theta, atol=1e-12, rtol=0.0))[0]

            lexicographic_used = len(leaving_candidates) > 1
            B_inv = self._compute_basis_inverse(P, L, U) if lexicographic_used else None
            if lexicographic_used:
                leaving_row = self._lexicographic_leaving_row(xb, B_inv, d, leaving_candidates)
            else:
                leaving_row = int(leaving_candidates[0])

            leaving_var = basis[leaving_row]

            state["theta"] = theta
            state["ratios"] = ratios.copy()
            state["leaving_row"] = leaving_row
            state["leaving_var"] = leaving_var
            state["lexicographic_used"] = lexicographic_used
            state["lexicographic_candidates"] = [int(x) for x in leaving_candidates.tolist()]
            if B_inv is not None:
                state["B_inv"] = B_inv.copy()
                state["message"] = (
                    "Degeneracy detected. Applying the Charnes rule for lexicographic tie-breaking. "
                    "The basis inverse was recovered from the current PLU factorization."
                )

            yield state

            # 7) Actualizare dupa pivot.
            xb = xb - theta * d
            xb[leaving_row] = theta
            xb[np.abs(xb) < self.tol] = 0.0
            basis[leaving_row] = entering

        raise ValueError(
            f"The maximum number of iterations ({self.max_iter}) was reached without a conclusion."
        )


class MatrixGrid(ttk.Frame):
    """Afisare tabelara aerisita pentru matrici si vectori, in stil Azure."""

    def __init__(self, parent, title):
        super().__init__(parent)
        self.title_label = tk.Label(
            self,
            text=title,
            bg="#1e3a8a",
            fg="#ffffff",
            padx=12,
            pady=7,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.title_label.pack(fill="x", pady=(0, 6))
        self.grid_frame = ttk.Frame(self)
        self.grid_frame.pack(fill="both", expand=True)
        self.widgets = []

    def clear(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.widgets = []

    def set_data(
        self,
        data,
        row_headers=None,
        col_headers=None,
        highlight_rows=None,
        highlight_cols=None,
        fmt="{:.4f}",
    ):
        self.clear()
        highlight_rows = set(highlight_rows or [])
        highlight_cols = set(highlight_cols or [])

        if data is None:
            tk.Label(
                self.grid_frame,
                text="—",
                bg="#ffffff",
                fg="#64748b",
                width=12,
                padx=8,
                pady=6,
                relief="solid",
                bd=1,
            ).grid(row=0, column=0, sticky="w")
            return

        arr = np.array(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        rows, cols = arr.shape
        if row_headers is None:
            row_headers = [str(i + 1) for i in range(rows)]
        if col_headers is None:
            col_headers = [str(j + 1) for j in range(cols)]

        def make_cell(r, c, text, bg="#ffffff", fg="#0f172a", bold=False):
            lbl = tk.Label(
                self.grid_frame,
                text=text,
                bg=bg,
                fg=fg,
                relief="solid",
                bd=1,
                padx=10,
                pady=7,
                width=12,
                font=("Segoe UI", 9, "bold" if bold else "normal"),
            )
            lbl.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            return lbl

        make_cell(0, 0, "", bg="#dbeafe", fg="#1e3a8a", bold=True)
        for j, header in enumerate(col_headers, start=1):
            bg = "#d9f2d9" if (j - 1) in highlight_cols else "#dbeafe"
            make_cell(0, j, header, bg=bg, fg="#1e3a8a", bold=True)

        for i in range(rows):
            row_bg = "#f8d7da" if i in highlight_rows else "#eff6ff"
            make_cell(i + 1, 0, row_headers[i], bg=row_bg, fg="#1e3a8a", bold=True)
            for j in range(cols):
                value = arr[i, j]
                fg = "#94a3b8" if abs(value) < 1e-9 else "#0f172a"
                cell_bg = "#ffffff"
                if i in highlight_rows and j in highlight_cols:
                    cell_bg = "#f6d5b5"
                elif i in highlight_rows:
                    cell_bg = "#fdecef"
                elif j in highlight_cols:
                    cell_bg = "#eaf8ea"
                make_cell(i + 1, j + 1, fmt.format(value), bg=cell_bg, fg=fg)

        for i in range(rows + 1):
            self.grid_frame.grid_rowconfigure(i, weight=1)
        for j in range(cols + 1):
            self.grid_frame.grid_columnconfigure(j, weight=1)


class SimplexInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Revised Simplex Pro V9 - Azure Edition")
        self.root.geometry("1560x980")
        self.root.minsize(1320, 820)
        self.root.configure(bg="#eff6ff")

        self.problem_type = tk.StringVar(value="max")
        self.n_var = tk.IntVar(value=2)
        self.m_constr = tk.IntVar(value=3)

        self.c_entries = []
        self.a_entries = []
        self.b_entries = []
        self.sign_boxes = []
        self.entry_positions = {}
        self.var_header_labels = []
        self.row_header_labels = []

        self.solver = None
        self.generator = None
        self.current_state = None
        self.final_result = None
        self.current_data_signature = None
        self.finished = False
        self.language = "en"
        self.language_button = None
        self.ai_windows = []
        self.ai_problem_context = ""
        # Păstrăm rândul de notare în cod, dar îl ascundem implicit ca să dispară din UI.
        self.show_cerc_note_row = False
        self._scrollable_canvases = []
        self._mousewheel_bound = False

        self._build_style()
        self._build_layout()
        self.generate_table()

    # ===================== Localizare EN / RO =====================
    def localize_text(self, text):
        return translate_text_for_language(text, getattr(self, "language", "en"))

    def _apply_language_to_widget_tree(self, widget):
        try:
            current_text = widget.cget("text")
            translated = self.localize_text(current_text)
            if translated != current_text:
                widget.configure(text=translated)
        except Exception:
            pass

        try:
            for child in widget.winfo_children():
                self._apply_language_to_widget_tree(child)
        except Exception:
            pass

    def _apply_language_to_notebook(self):
        try:
            total_tabs = self.notebook.index("end")
            for idx in range(total_tabs):
                current_text = self.notebook.tab(idx, "text")
                translated = self.localize_text(current_text)
                if translated != current_text:
                    self.notebook.tab(idx, text=translated)
        except Exception:
            pass

    def _apply_language_to_trees(self):
        for tree_name in ("variables_tree", "constraints_tree"):
            tree = getattr(self, tree_name, None)
            if tree is None:
                continue
            try:
                for col in tree["columns"]:
                    current_text = tree.heading(col, "text")
                    translated = self.localize_text(current_text)
                    if translated != current_text:
                        tree.heading(col, text=translated)
                for item in tree.get_children():
                    current_values = tree.item(item, "values")
                    translated_values = tuple(self.localize_text(str(value)) for value in current_values)
                    if tuple(current_values) != translated_values:
                        tree.item(item, values=translated_values)
            except Exception:
                pass

    def _apply_language_to_plot_labels(self):
        try:
            self.ax.set_title(self.localize_text("Monotone convergence of Z_ext = c_B^T x_B"))
            self.ax.set_xlabel(self.localize_text("Iteration"))
            self.ax.set_ylabel(self.localize_text("Z_ext = c_B^T x_B"))
            self.canvas_plot.draw_idle()
        except Exception:
            pass

    def _translate_text_widget_content(self, widget):
        """Translate already-rendered Text/ScrolledText content when the language is toggled."""
        try:
            old_state = str(widget.cget("state"))
            widget.configure(state="normal")
            current = widget.get("1.0", "end-1c")
            translated = self.localize_text(current)
            if translated != current:
                widget.delete("1.0", tk.END)
                widget.insert(tk.END, translated)
            widget.configure(state=old_state)
        except Exception:
            pass

    def _apply_language_to_text_widgets(self):
        for name in ("preview_text", "log_text"):
            widget = getattr(self, name, None)
            if widget is not None:
                self._translate_text_widget_content(widget)

    def _apply_language(self):
        self.root.title(self.localize_text("Revised Simplex Pro V9 - Azure Edition"))
        self._apply_language_to_widget_tree(self.root)
        self._apply_language_to_notebook()
        self._apply_language_to_trees()
        self._apply_language_to_text_widgets()
        self._apply_language_to_plot_labels()
        self._refresh_config_controls_canvas()
        if self.language_button is not None:
            self.language_button.configure(text="EN" if self.language == "ro" else "RO")

        live_windows = []
        for assistant_window in getattr(self, "ai_windows", []):
            try:
                if assistant_window.window.winfo_exists():
                    assistant_window.apply_language()
                    live_windows.append(assistant_window)
            except Exception:
                pass
        self.ai_windows = live_windows

    def toggle_language(self):
        self.language = "ro" if self.language == "en" else "en"
        self._apply_language()

    # ===================== Stil si layout =====================
    def _build_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        azure_dark = "#1e3a8a"
        azure = "#2563eb"
        azure_mid = "#3b82f6"
        azure_soft = "#dbeafe"
        azure_bg = "#eff6ff"
        slate_text = "#334155"

        style.configure("TFrame", background=azure_bg)
        style.configure("TLabelframe", background=azure_bg, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=azure_bg, foreground=azure_dark, font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=azure_bg, foreground=slate_text)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=azure_dark, background=azure_bg)
        style.configure("PanelTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground=azure_dark, background=azure_bg)
        style.configure("ResultBig.TLabel", font=("Segoe UI", 22, "bold"), foreground=azure, background=azure_bg)
        style.configure("Warn.TLabel", foreground="#b02a37", font=("Segoe UI", 11, "bold"), background=azure_bg)
        style.configure("Info.TLabel", foreground=slate_text, background=azure_bg)

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background=azure,
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 7),
        )
        style.map(
            "Accent.TButton",
            background=[("active", azure_dark), ("pressed", azure_dark)],
            foreground=[("disabled", "#dbeafe")],
        )
        style.configure(
            "Json.TButton",
            font=("Segoe UI", 10, "bold"),
            background=azure_dark,
            foreground="#ffffff",
            borderwidth=0,
            padding=(10, 7),
        )
        style.map(
            "Json.TButton",
            background=[("active", "#1d4ed8"), ("pressed", "#1d4ed8")],
            foreground=[("disabled", "#dbeafe")],
        )
        style.configure(
            "Soft.TButton",
            font=("Segoe UI", 9),
            background=azure_soft,
            foreground=azure_dark,
            borderwidth=1,
            padding=(8, 6),
        )
        style.map("Soft.TButton", background=[("active", "#bfdbfe")])

        style.configure("TNotebook", background=azure_bg, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
            background="#dbeafe",
            foreground=azure_dark,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#ffffff"), ("active", "#bfdbfe")],
            foreground=[("selected", azure_dark)],
        )

        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")
        style.configure("Treeview", rowheight=24, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background=azure_soft, foreground=azure_dark)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)
        self.notebook = notebook

        self.tab_input = ttk.Frame(notebook, padding=10)
        self.tab_process = ttk.Frame(notebook, padding=10)
        self.tab_result = ttk.Frame(notebook, padding=10)
        self.tab_verify = ttk.Frame(notebook, padding=10)

        notebook.add(self.tab_input, text="1. Setup & Input")
        notebook.add(self.tab_process, text="2. Iteration Process")
        notebook.add(self.tab_result, text="3. Results & Analysis")
        notebook.add(self.tab_verify, text="4. Verification & Charts")

        self._build_tab_input()
        self._build_tab_process()
        self._build_tab_result()
        self._build_tab_verify()

    def _build_tab_input(self):
        self.tab_input.columnconfigure(0, weight=1)
        self.tab_input.rowconfigure(1, weight=1)

        # Bara de configurare poate deveni mai lată decât fereastra, mai ales în română.
        # O punem într-un Canvas cu scrollbar orizontal ca să nu dispară butoanele din dreapta.
        top_outer = ttk.LabelFrame(self.tab_input, text="Problem setup", padding=6)
        top_outer.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        top_outer.columnconfigure(0, weight=1)

        self.config_canvas = tk.Canvas(top_outer, background="#eff6ff", highlightthickness=0, height=48)
        self.config_canvas.grid(row=0, column=0, sticky="ew")
        self.config_hbar = ttk.Scrollbar(top_outer, orient="horizontal", command=self.config_canvas.xview)
        self.config_hbar.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.config_canvas.configure(xscrollcommand=self.config_hbar.set)
        self._register_scrollable_canvas(self.config_canvas)

        top = ttk.Frame(self.config_canvas, padding=(4, 4, 4, 4))
        self.config_inner = top
        self.config_window = self.config_canvas.create_window((0, 0), window=top, anchor="nw")
        top.bind("<Configure>", self._on_config_controls_configure)
        self.config_canvas.bind("<Configure>", self._on_config_canvas_configure)

        ttk.Label(top, text="Objective type:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        ttk.Combobox(top, textvariable=self.problem_type, values=["max", "min"], width=8, state="readonly").grid(
            row=0, column=1, padx=5, pady=4, sticky="w"
        )

        ttk.Label(top, text="Number of variables (n):").grid(row=0, column=2, padx=5, pady=4, sticky="w")
        ttk.Entry(top, textvariable=self.n_var, width=7).grid(row=0, column=3, padx=5, pady=4, sticky="w")

        ttk.Label(top, text="Number of constraints (m):").grid(row=0, column=4, padx=5, pady=4, sticky="w")
        ttk.Entry(top, textvariable=self.m_constr, width=7).grid(row=0, column=5, padx=5, pady=4, sticky="w")

        ttk.Button(top, text="Load JSON", command=self.load_from_json, style="Json.TButton").grid(
            row=0, column=6, padx=(14, 6), pady=4, sticky="w"
        )
        ttk.Button(top, text="Save JSON", command=self.save_to_json, style="Json.TButton").grid(
            row=0, column=7, padx=6, pady=4, sticky="w"
        )
        ttk.Button(top, text="Generate table", command=self.generate_table, style="Soft.TButton").grid(
            row=0, column=8, padx=6, pady=4, sticky="w"
        )
        ttk.Button(top, text="Load example", command=self.load_example, style="Soft.TButton").grid(
            row=0, column=9, padx=6, pady=4, sticky="w"
        )
        ttk.Button(top, text="Next step", style="Accent.TButton", command=self.next_step).grid(
            row=0, column=10, padx=6, pady=4, sticky="w"
        )
        ttk.Button(top, text="Solve completely", style="Accent.TButton", command=self.solve_complete).grid(
            row=0, column=11, padx=6, pady=4, sticky="w"
        )
        ttk.Button(top, text="Reset run", command=self.reset_run_state, style="Soft.TButton").grid(
            row=0, column=12, padx=6, pady=4, sticky="w"
        )

        ttk.Button(
            top,
            text="AI Assistant",
            command=self.open_ai_assistant,
            style="Accent.TButton"
        ).grid(
            row=0,
            column=13,
            padx=6,
            pady=4,
            sticky="w"
        )

        self.language_button = ttk.Button(
            top,
            text="RO",
            command=self.toggle_language,
            style="Soft.TButton"
        )
        self.language_button.grid(
            row=0,
            column=14,
            padx=6,
            pady=4,
            sticky="w"
        )

        self._refresh_config_controls_canvas()

        main = ttk.Frame(self.tab_input)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=4)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        # Zona scrollabila mare pentru input.
        input_box = ttk.LabelFrame(main, text="Problem matrix", padding=8)
        input_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        input_box.rowconfigure(0, weight=1)
        input_box.columnconfigure(0, weight=1)

        self.input_canvas = tk.Canvas(input_box, background="#eff6ff", highlightthickness=0)
        self.input_canvas.grid(row=0, column=0, sticky="nsew")
        ybar = ttk.Scrollbar(input_box, orient="vertical", command=self.input_canvas.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        xbar = ttk.Scrollbar(input_box, orient="horizontal", command=self.input_canvas.xview)
        xbar.grid(row=1, column=0, sticky="ew")
        self.input_canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        self._register_scrollable_canvas(self.input_canvas)

        self.input_inner = ttk.Frame(self.input_canvas, padding=6)
        self.input_window = self.input_canvas.create_window((0, 0), window=self.input_inner, anchor="nw")
        self.input_inner.bind("<Configure>", self._on_input_configure)
        self.input_canvas.bind("<Configure>", self._on_canvas_configure)

        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="nsew")
        side.rowconfigure(1, weight=1)
        side.rowconfigure(2, weight=1)
        side.columnconfigure(0, weight=1)

        guide = ttk.LabelFrame(side, text="Help", padding=8)
        guide.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        help_text = (
            "• Use <=, >=, or = for each constraint.\n"
            "• Navigation: arrow keys + Enter between cells.\n"
            "• If b is negative, the solver automatically normalizes the row.\n"
            "• Artificial variables are handled with the Big-M method."
        )
        ttk.Label(guide, text=help_text, style="Info.TLabel", justify="left").pack(anchor="w")

        std_box = ttk.LabelFrame(side, text="Current state preview", padding=8)
        std_box.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        std_box.columnconfigure(0, weight=1)
        std_box.rowconfigure(1, weight=1)
        self.preview_label = ttk.Label(std_box, text="No iteration has been run yet.", style="Info.TLabel")
        self.preview_label.grid(row=0, column=0, sticky="w")
        self.preview_text = ScrolledText(std_box, height=12, wrap="word", font=("Consolas", 9))
        self.preview_text.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.preview_text.configure(state="disabled")

        log_box = ttk.LabelFrame(side, text="Technical log", padding=8)
        log_box.grid(row=2, column=0, sticky="nsew")
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_box, height=14, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

    def _build_tab_process(self):
        self.tab_process.rowconfigure(0, weight=1)
        self.tab_process.columnconfigure(0, weight=1)

        self.process_canvas = tk.Canvas(self.tab_process, background="#eff6ff", highlightthickness=0)
        self.process_canvas.grid(row=0, column=0, sticky="nsew")
        process_vbar = ttk.Scrollbar(self.tab_process, orient="vertical", command=self.process_canvas.yview)
        process_vbar.grid(row=0, column=1, sticky="ns")
        process_hbar = ttk.Scrollbar(self.tab_process, orient="horizontal", command=self.process_canvas.xview)
        process_hbar.grid(row=1, column=0, sticky="ew")
        self.process_canvas.configure(yscrollcommand=process_vbar.set, xscrollcommand=process_hbar.set)
        self._register_scrollable_canvas(self.process_canvas)

        self.process_inner = ttk.Frame(self.process_canvas, padding=10)
        self.process_window = self.process_canvas.create_window((0, 0), window=self.process_inner, anchor="nw")
        self.process_inner.bind("<Configure>", self._on_process_configure)
        self.process_canvas.bind("<Configure>", self._on_process_canvas_configure)

        self.process_inner.columnconfigure(0, weight=3)
        self.process_inner.columnconfigure(1, weight=2)
        self.process_inner.rowconfigure(1, weight=1)

        summary = ttk.LabelFrame(self.process_inner, text="Iteration summary", padding=10)
        summary.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        summary.columnconfigure(0, weight=1)
        self.iteration_summary = ttk.Label(summary, text="No iteration data available yet.", style="Info.TLabel")
        self.iteration_summary.grid(row=0, column=0, sticky="w")

        left = ttk.Frame(self.process_inner)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        upper = ttk.Frame(left)
        upper.grid(row=0, column=0, sticky="nsew")
        upper.columnconfigure(0, weight=1)
        upper.columnconfigure(1, weight=1)
        upper.rowconfigure(0, weight=1)

        lower = ttk.Frame(left)
        lower.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1)
        lower.rowconfigure(0, weight=1)

        self.table_B = MatrixGrid(upper, "Current basis B")
        self.table_B.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.table_L = MatrixGrid(upper, "Lower triangular factor L")
        self.table_L.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.table_U = MatrixGrid(lower, "Upper triangular factor U")
        self.table_U.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.table_Aext = MatrixGrid(lower, "Standardized matrix A")
        self.table_Aext.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        right = ttk.Frame(self.process_inner)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        self.table_pi = MatrixGrid(right, "Simplex multiplier π")
        self.table_pi.grid(row=0, column=0, sticky="nsew")
        self.table_reduced = MatrixGrid(right, "Costuri reduse Δ = c - z")
        self.table_reduced.grid(row=1, column=0, sticky="nsew", pady=(12, 12))
        self.table_direction = MatrixGrid(right, "Direction d = B⁻¹P_k / Ratios θ_i")
        self.table_direction.grid(row=2, column=0, sticky="nsew")

    def _build_tab_result(self):
        self.tab_result.columnconfigure(0, weight=2)
        self.tab_result.columnconfigure(1, weight=2)
        self.tab_result.rowconfigure(1, weight=1)

        top = ttk.Frame(self.tab_result)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Final value f(x*):", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.z_big_label = ttk.Label(top, text="—", style="ResultBig.TLabel")
        self.z_big_label.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.result_status_label = ttk.Label(top, text="No completed solve yet.", style="Info.TLabel")
        self.result_status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        vars_box = ttk.LabelFrame(self.tab_result, text="Solution vector x*", padding=8)
        vars_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        vars_box.columnconfigure(0, weight=1)
        vars_box.rowconfigure(0, weight=1)
        self.variables_tree = ttk.Treeview(
            vars_box,
            columns=("variable", "tip", "valoare"),
            show="headings",
        )
        self.variables_tree.heading("variable", text="Variable")
        self.variables_tree.heading("tip", text="Type")
        self.variables_tree.heading("valoare", text="Value")
        self.variables_tree.column("variable", width=120, anchor="w")
        self.variables_tree.column("tip", width=140, anchor="center")
        self.variables_tree.column("valoare", width=140, anchor="e")
        self.variables_tree.grid(row=0, column=0, sticky="nsew")
        vars_scroll = ttk.Scrollbar(vars_box, orient="vertical", command=self.variables_tree.yview)
        vars_scroll.grid(row=0, column=1, sticky="ns")
        self.variables_tree.configure(yscrollcommand=vars_scroll.set)

        res_box = ttk.LabelFrame(self.tab_result, text="Constraint check Ax* ? b", padding=8)
        res_box.grid(row=1, column=1, sticky="nsew")
        res_box.columnconfigure(0, weight=1)
        res_box.rowconfigure(0, weight=1)
        self.constraints_tree = ttk.Treeview(
            res_box,
            columns=("constraint", "lhs", "sign", "rhs", "status"),
            show="headings",
        )
        for col, text, width in [
            ("constraint", "Constraint", 90),
            ("lhs", "LHS", 110),
            ("sign", "Sign", 70),
            ("rhs", "RHS", 110),
            ("status", "Status", 120),
        ]:
            self.constraints_tree.heading(col, text=text)
            self.constraints_tree.column(col, width=width, anchor="center")
        self.constraints_tree.grid(row=0, column=0, sticky="nsew")
        cscroll = ttk.Scrollbar(res_box, orient="vertical", command=self.constraints_tree.yview)
        cscroll.grid(row=0, column=1, sticky="ns")
        self.constraints_tree.configure(yscrollcommand=cscroll.set)

    def _build_tab_verify(self):
        self.tab_verify.columnconfigure(0, weight=1)
        self.tab_verify.rowconfigure(1, weight=1)
        self.tab_verify.rowconfigure(2, weight=2)

        info = ttk.LabelFrame(self.tab_verify, text="Mathematical verification", padding=8)
        info.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.verify_label = ttk.Label(info, text="No verification available.", style="Info.TLabel")
        self.verify_label.pack(anchor="w")

        tables = ttk.Frame(self.tab_verify)
        tables.grid(row=1, column=0, sticky="nsew")
        tables.columnconfigure(0, weight=1)
        tables.columnconfigure(1, weight=1)
        tables.rowconfigure(0, weight=1)

        self.table_verify_original = MatrixGrid(tables, "Verification A · x vs b")
        self.table_verify_original.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.table_verify_standard = MatrixGrid(tables, "Verification A_ext · x_ext vs b")
        self.table_verify_standard.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        chart_box = ttk.LabelFrame(self.tab_verify, text="Z convergence chart by iteration", padding=8)
        chart_box.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        chart_box.rowconfigure(0, weight=1)
        chart_box.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(6, 3.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Monotone convergence of Z_ext = c_B^T x_B")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Z")
        self.figure.tight_layout()
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=chart_box)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ===================== Scroll / input helpers =====================
    def _register_scrollable_canvas(self, canvas):
        """Register canvases that need reliable vertical and horizontal scrolling."""
        if canvas not in self._scrollable_canvases:
            self._scrollable_canvases.append(canvas)
        if not self._mousewheel_bound:
            self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")
            self.root.bind_all("<Shift-MouseWheel>", self._on_global_shift_mousewheel, add="+")
            self.root.bind_all("<Button-4>", self._on_global_mousewheel_linux, add="+")
            self.root.bind_all("<Button-5>", self._on_global_mousewheel_linux, add="+")
            self.root.bind_all("<Shift-Button-4>", self._on_global_shift_mousewheel_linux, add="+")
            self.root.bind_all("<Shift-Button-5>", self._on_global_shift_mousewheel_linux, add="+")
            self._mousewheel_bound = True

    def _canvas_under_pointer(self):
        try:
            pointer_x = self.root.winfo_pointerx()
            pointer_y = self.root.winfo_pointery()
            for canvas in self._scrollable_canvases:
                if not canvas.winfo_exists():
                    continue
                x0 = canvas.winfo_rootx()
                y0 = canvas.winfo_rooty()
                x1 = x0 + canvas.winfo_width()
                y1 = y0 + canvas.winfo_height()
                if x0 <= pointer_x <= x1 and y0 <= pointer_y <= y1:
                    return canvas
        except Exception:
            return None
        return None

    def _wheel_units(self, event):
        delta = getattr(event, "delta", 0)
        if delta:
            units = int(-delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            return units or (-1 if delta > 0 else 1)
        return -1 if getattr(event, "num", None) == 4 else 1

    def _on_global_mousewheel(self, event):
        canvas = self._canvas_under_pointer()
        if canvas is None:
            return None
        if canvas is getattr(self, "config_canvas", None):
            canvas.xview_scroll(self._wheel_units(event), "units")
        elif getattr(event, "state", 0) & 0x0001:
            canvas.xview_scroll(self._wheel_units(event), "units")
        else:
            canvas.yview_scroll(self._wheel_units(event), "units")
        return "break"

    def _on_global_shift_mousewheel(self, event):
        canvas = self._canvas_under_pointer()
        if canvas is None:
            return None
        canvas.xview_scroll(self._wheel_units(event), "units")
        return "break"

    def _on_global_mousewheel_linux(self, event):
        canvas = self._canvas_under_pointer()
        if canvas is None:
            return None
        canvas.yview_scroll(self._wheel_units(event), "units")
        return "break"

    def _on_global_shift_mousewheel_linux(self, event):
        canvas = self._canvas_under_pointer()
        if canvas is None:
            return None
        canvas.xview_scroll(self._wheel_units(event), "units")
        return "break"

    def _set_canvas_window_size(self, canvas, window, width, height):
        try:
            current_width = int(float(canvas.itemcget(window, "width") or 0))
            current_height = int(float(canvas.itemcget(window, "height") or 0))
        except Exception:
            current_width = current_height = 0
        if abs(current_width - int(width)) > 1 or abs(current_height - int(height)) > 1:
            canvas.itemconfigure(window, width=int(width), height=int(height))

    def _refresh_scrollable_canvas(self, canvas, inner, window):
        """Keep scrollregion in sync so bottom horizontal sliders remain usable."""
        try:
            inner.update_idletasks()
            view_width = max(canvas.winfo_width(), 1)
            view_height = max(canvas.winfo_height(), 1)
            required_width = max(inner.winfo_reqwidth(), inner.winfo_width(), 1)
            required_height = max(inner.winfo_reqheight(), inner.winfo_height(), 1)
            scroll_width = max(required_width, view_width)
            scroll_height = max(required_height, view_height)
            self._set_canvas_window_size(canvas, window, scroll_width, scroll_height)
            canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        except Exception:
            pass

    def _refresh_config_controls_canvas(self):
        """Keep the top problem-setup toolbar horizontally scrollable."""
        try:
            canvas = self.config_canvas
            inner = self.config_inner
            window = self.config_window
            inner.update_idletasks()
            view_width = max(canvas.winfo_width(), 1)
            required_width = max(inner.winfo_reqwidth(), 1)
            required_height = max(inner.winfo_reqheight(), 1)
            canvas.itemconfigure(window, width=max(required_width, view_width), height=required_height)
            canvas.configure(scrollregion=(0, 0, required_width, required_height))
            canvas.configure(height=required_height)
        except Exception:
            pass

    def _on_config_controls_configure(self, _event=None):
        self._refresh_config_controls_canvas()

    def _on_config_canvas_configure(self, _event=None):
        self._refresh_config_controls_canvas()

    def _on_input_configure(self, _event=None):
        self._refresh_scrollable_canvas(self.input_canvas, self.input_inner, self.input_window)

    def _on_canvas_configure(self, _event=None):
        self._refresh_scrollable_canvas(self.input_canvas, self.input_inner, self.input_window)

    def _on_process_configure(self, _event=None):
        self._refresh_scrollable_canvas(self.process_canvas, self.process_inner, self.process_window)

    def _on_process_canvas_configure(self, _event=None):
        self._refresh_scrollable_canvas(self.process_canvas, self.process_inner, self.process_window)

    def _validate_numeric(self, value):
        if value == "":
            return True
        value = value.replace(",", ".")
        if value in {"-", "+", ".", "-.", "+."}:
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _clear_entry_highlights(self):
        for entry in self.c_entries:
            entry.configure(bg="#ffffff")
        for row in self.a_entries:
            for entry in row:
                entry.configure(bg="#ffffff")
        for entry in self.b_entries:
            entry.configure(bg="#f8fafc")
        for lbl in self.var_header_labels:
            lbl.configure(bg="#dbeafe", fg="#1e3a8a")
        for lbl in self.row_header_labels:
            lbl.configure(bg="#dbeafe", fg="#1e3a8a")

    def _apply_input_highlights(self, entering=None, leaving_row=None):
        self._clear_entry_highlights()
        n = int(self.n_var.get())
        if entering is not None and entering < n:
            self.var_header_labels[entering].configure(bg="#d9f2d9")
            for i in range(len(self.a_entries)):
                self.a_entries[i][entering].configure(bg="#eaf8ea")
            if entering < len(self.c_entries):
                self.c_entries[entering].configure(bg="#eaf8ea")

        if leaving_row is not None and 0 <= leaving_row < len(self.a_entries):
            self.row_header_labels[leaving_row].configure(bg="#f8d7da")
            for widget in self.a_entries[leaving_row]:
                widget.configure(bg="#fdecef")
            self.b_entries[leaving_row].configure(bg="#fdecef")

    def _bind_entry_navigation(self, widget, row_idx, col_idx):
        self.entry_positions[widget] = (row_idx, col_idx)
        widget.bind("<Up>", self._move_focus)
        widget.bind("<Down>", self._move_focus)
        widget.bind("<Left>", self._move_focus)
        widget.bind("<Right>", self._move_focus)
        widget.bind("<Return>", self._move_focus)

    def _scroll_widget_into_view(self, widget):
        self.root.update_idletasks()
        self._refresh_scrollable_canvas(self.input_canvas, self.input_inner, self.input_window)
        try:
            wx = widget.winfo_rootx() - self.input_inner.winfo_rootx()
            wy = widget.winfo_rooty() - self.input_inner.winfo_rooty()
            ww = max(widget.winfo_width(), 1)
            wh = max(widget.winfo_height(), 1)
            total_w = max(self.input_inner.winfo_reqwidth(), 1)
            total_h = max(self.input_inner.winfo_reqheight(), 1)
            view_w = max(self.input_canvas.winfo_width(), 1)
            view_h = max(self.input_canvas.winfo_height(), 1)

            target_left = max(0.0, min(float(total_w - view_w), wx + ww / 2 - view_w / 2))
            target_top = max(0.0, min(float(total_h - view_h), wy + wh / 2 - view_h / 2))

            self.input_canvas.xview_moveto(0.0 if total_w <= view_w else target_left / total_w)
            self.input_canvas.yview_moveto(0.0 if total_h <= view_h else target_top / total_h)
        except Exception:
            pass

    def _focus_entry(self, row_idx, col_idx):
        for widget, pos in self.entry_positions.items():
            if pos == (row_idx, col_idx):
                widget.focus_set()
                widget.icursor(tk.END)
                self._scroll_widget_into_view(widget)
                return True
        return False

    def _move_focus(self, event):
        if event.widget not in self.entry_positions:
            return None
        row_idx, col_idx = self.entry_positions[event.widget]
        target = (row_idx, col_idx)
        key = event.keysym

        if key == "Up":
            target = (row_idx - 1, col_idx)
        elif key in {"Down", "Return"}:
            target = (row_idx + 1, col_idx)
        elif key == "Left":
            target = (row_idx, col_idx - 1)
        elif key == "Right":
            target = (row_idx, col_idx + 1)

        if self._focus_entry(*target):
            return "break"
        return None

    # ===================== Tabel input =====================
    def generate_table(self):
        try:
            n = int(self.n_var.get())
            m = int(self.m_constr.get())
            if n <= 0 or m <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid data", "The number of variables and constraints must be positive integers.")
            return

        for child in self.input_inner.winfo_children():
            child.destroy()

        self.c_entries = []
        self.a_entries = []
        self.b_entries = []
        self.sign_boxes = []
        self.entry_positions = {}
        self.var_header_labels = []
        self.row_header_labels = []
        self.reset_run_state(clear_inputs=False, preserve_log=False)

        vcmd = (self.root.register(self._validate_numeric), "%P")

        ttk.Label(self.input_inner, text="Input for the revised simplex method", style="Title.TLabel").grid(
            row=0, column=0, columnspan=n + 3, sticky="w", pady=(0, 8)
        )

        # Rândul informativ CERC rămâne disponibil în cod, dar este ascuns implicit.
        # Dacă vrei să îl reactivezi, setează self.show_cerc_note_row = True.
        content_row_offset = 0
        if getattr(self, "show_cerc_note_row", False):
            ttk.Label(
                self.input_inner,
                text="CERC notation: x for variables, y for slack/surplus variables, z for artificial variables; keyboard navigation and JSON persistence.",
                style="Info.TLabel",
            ).grid(row=1, column=0, columnspan=n + 3, sticky="w", pady=(0, 12))
            content_row_offset = 1

        objective_row = 1 + content_row_offset
        separator_row = objective_row + 1
        system_title_row = separator_row + 1
        header_row = system_title_row + 1
        first_constraint_row = header_row + 1

        ttk.Label(self.input_inner, text="Objective function f(x) =", style="PanelTitle.TLabel").grid(
            row=objective_row, column=0, padx=6, pady=(2, 10), sticky="w"
        )
        for j in range(n):
            cell = ttk.Frame(self.input_inner)
            cell.grid(row=objective_row, column=j + 1, padx=6, pady=(0, 10), sticky="n")
            lbl = tk.Label(
                cell,
                text=f"x{j + 1}",
                bg="#dbeafe",
                fg="#1e3a8a",
                relief="flat",
                padx=10,
                pady=4,
                font=("Segoe UI", 9, "bold"),
            )
            lbl.pack(fill="x", pady=(0, 3))
            self.var_header_labels.append(lbl)

            entry = tk.Entry(
                cell,
                width=10,
                justify="center",
                bg="#ffffff",
                relief="solid",
                bd=1,
                validate="key",
                validatecommand=vcmd,
            )
            entry.pack()
            self.c_entries.append(entry)
            self._bind_entry_navigation(entry, 0, j)

        ttk.Separator(self.input_inner, orient="horizontal").grid(
            row=separator_row, column=0, columnspan=n + 3, sticky="ew", pady=(6, 12)
        )
        ttk.Label(self.input_inner, text="System Ax = b after standardization", style="PanelTitle.TLabel").grid(
            row=system_title_row, column=0, columnspan=n + 3, sticky="w", pady=(0, 6)
        )

        row_header = tk.Label(
            self.input_inner,
            text="Row i",
            bg="#dbeafe",
            fg="#1e3a8a",
            relief="flat",
            padx=8,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        )
        row_header.grid(row=header_row, column=0, padx=2, pady=2, sticky="nsew")

        for j in range(n):
            lbl = tk.Label(
                self.input_inner,
                text=f"x{j + 1}",
                bg="#dbeafe",
                fg="#1e3a8a",
                relief="flat",
                padx=8,
                pady=6,
                font=("Segoe UI", 9, "bold"),
            )
            lbl.grid(row=header_row, column=j + 1, padx=2, pady=2, sticky="nsew")
        tk.Label(
            self.input_inner,
            text="Sign",
            bg="#dbeafe",
            fg="#1e3a8a",
            relief="flat",
            padx=8,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=header_row, column=n + 1, padx=2, pady=2, sticky="nsew")
        tk.Label(
            self.input_inner,
            text="b",
            bg="#dbeafe",
            fg="#1e3a8a",
            relief="flat",
            padx=8,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=header_row, column=n + 2, padx=2, pady=2, sticky="nsew")

        for i in range(m):
            row = first_constraint_row + i
            row_lbl = tk.Label(
                self.input_inner,
                text=f"R{i + 1}",
                bg="#dbeafe",
                fg="#1e3a8a",
                relief="flat",
                padx=8,
                pady=6,
                font=("Segoe UI", 9, "bold"),
            )
            row_lbl.grid(row=row, column=0, padx=2, pady=3, sticky="nsew")
            self.row_header_labels.append(row_lbl)

            a_row = []
            for j in range(n):
                entry = tk.Entry(
                    self.input_inner,
                    width=10,
                    justify="center",
                    bg="#ffffff",
                    relief="solid",
                    bd=1,
                    validate="key",
                    validatecommand=vcmd,
                )
                entry.grid(row=row, column=j + 1, padx=3, pady=3, sticky="nsew")
                a_row.append(entry)
                self._bind_entry_navigation(entry, i + 1, j)
            self.a_entries.append(a_row)

            sign_box = ttk.Combobox(self.input_inner, values=["<=", ">=", "="], width=6, state="readonly")
            sign_box.set("<=")
            sign_box.grid(row=row, column=n + 1, padx=4, pady=3)
            self.sign_boxes.append(sign_box)

            b_entry = tk.Entry(
                self.input_inner,
                width=10,
                justify="center",
                bg="#f8fafc",
                relief="solid",
                bd=1,
                validate="key",
                validatecommand=vcmd,
            )
            b_entry.grid(row=row, column=n + 2, padx=3, pady=3, sticky="nsew")
            self.b_entries.append(b_entry)
            self._bind_entry_navigation(b_entry, i + 1, n)

        note = (
            "Note: standardization introduces y variables for slack/surplus and z variables for artificial variables penalized by M."
        )
        ttk.Label(self.input_inner, text=note, style="Info.TLabel").grid(
            row=first_constraint_row + m, column=0, columnspan=n + 3, sticky="w", pady=(12, 0)
        )

        self._refresh_scrollable_canvas(self.input_canvas, self.input_inner, self.input_window)
        self._on_input_configure()
        if self.c_entries:
            self.c_entries[0].focus_set()
            self._scroll_widget_into_view(self.c_entries[0])
        if getattr(self, "language", "en") != "en":
            self._apply_language()

    # ===================== Colectare si validare date =====================
    def _entry_value(self, entry, label):
        raw = entry.get().strip().replace(",", ".")
        if raw == "":
            raise ValueError(f"Campul {label} este gol.")
        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(f"Campul {label} trebuie sa contina un numar valid.") from exc

    def collect_data(self):
        n = int(self.n_var.get())
        m = int(self.m_constr.get())

        c = [self._entry_value(self.c_entries[j], f"c{j + 1}") for j in range(n)]
        A = []
        for i in range(m):
            row = [self._entry_value(self.a_entries[i][j], f"A[{i + 1},{j + 1}]") for j in range(n)]
            A.append(row)
        b = [self._entry_value(self.b_entries[i], f"b{i + 1}") for i in range(m)]
        signs = [box.get().strip() for box in self.sign_boxes]

        signature = (
            self.problem_type.get().strip(),
            tuple(c),
            tuple(tuple(row) for row in A),
            tuple(b),
            tuple(signs),
        )
        return c, A, b, signs, signature

    # ===================== Exemplu si stare rulare =====================
    def clear_text_widget(self, widget):
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.configure(state="disabled")

    def set_text_widget(self, widget, text):
        if getattr(self, "language", "en") == "ro":
            text = self.localize_text(text)
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def append_log(self, text):
        if getattr(self, "language", "en") == "ro":
            text = self.localize_text(text)
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")
        self.root.update_idletasks()

    def reset_run_state(self, clear_inputs=False, preserve_log=True):
        self.generator = None
        self.solver = None
        self.current_state = None
        self.final_result = None
        self.current_data_signature = None
        self.finished = False

        if not preserve_log:
            self.clear_text_widget(self.log_text)
        self._clear_entry_highlights()
        self.preview_label.configure(text="No iteration has been run yet.")
        self.set_text_widget(self.preview_text, "")
        self.iteration_summary.configure(text="No iteration data available yet.")
        self.z_big_label.configure(text="—")
        self.result_status_label.configure(text="No completed solve yet.", style="Info.TLabel")
        self.verify_label.configure(text="No verification available.")
        self._clear_tree(self.variables_tree)
        self._clear_tree(self.constraints_tree)
        self.table_B.set_data(None)
        self.table_L.set_data(None)
        self.table_U.set_data(None)
        self.table_Aext.set_data(None)
        self.table_pi.set_data(None)
        self.table_reduced.set_data(None)
        self.table_direction.set_data(None)
        self.table_verify_original.set_data(None)
        self.table_verify_standard.set_data(None)
        self._update_plot([])

        if clear_inputs:
            for entry in self.c_entries:
                entry.delete(0, tk.END)
            for row in self.a_entries:
                for entry in row:
                    entry.delete(0, tk.END)
            for entry in self.b_entries:
                entry.delete(0, tk.END)
            for box in self.sign_boxes:
                box.set("<=")

        if getattr(self, "language", "en") != "en":
            self._apply_language()

    def _collect_form_snapshot(self):
        return {
            "problem_type": self.problem_type.get().strip(),
            "n": int(self.n_var.get()),
            "m": int(self.m_constr.get()),
            "c": [entry.get() for entry in self.c_entries],
            "A": [[entry.get() for entry in row] for row in self.a_entries],
            "b": [entry.get() for entry in self.b_entries],
            "signs": [box.get().strip() for box in self.sign_boxes],
        }

    def save_to_json(self):
        try:
            data = self._collect_form_snapshot()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        path = filedialog.asksaveasfilename(
            title="Save problem",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        self.append_log(f"Problem saved to: {path}")
        messagebox.showinfo("Save successful", "The problem was saved as JSON.")

    def load_from_json(self):
        path = filedialog.askopenfilename(
            title="Load problem",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            n = int(data["n"])
            m = int(data["m"])
            problem_type = str(data.get("problem_type", data.get("type", data.get("obj", "max")))).strip().lower()
            if problem_type not in {"max", "min"}:
                raise ValueError("The 'problem_type' field must be 'max' or 'min'.")

            c_vals = data["c"]
            A_vals = data["A"] if "A" in data else data["a"]
            b_vals = data["b"]
            signs_vals = data["signs"]

            if len(c_vals) != n:
                raise ValueError("The size of vector c does not match n.")
            if len(A_vals) != m or any(len(row) != n for row in A_vals):
                raise ValueError("The size of matrix A does not match m and n.")
            if len(b_vals) != m or len(signs_vals) != m:
                raise ValueError("Vectors b and signs must have exactly m elements.")

            self.problem_type.set(problem_type)
            self.n_var.set(n)
            self.m_constr.set(m)
            self.generate_table()

            for j, value in enumerate(c_vals):
                self.c_entries[j].insert(0, str(value))
            for i in range(m):
                for j in range(n):
                    self.a_entries[i][j].insert(0, str(A_vals[i][j]))
                self.b_entries[i].insert(0, str(b_vals[i]))
                sign = str(signs_vals[i]).strip()
                self.sign_boxes[i].set(sign if sign in {"<=", ">=", "="} else "<=")

            self.clear_text_widget(self.log_text)
            self.append_log(f"Problem loaded from: {path}")
            messagebox.showinfo("Load successful", "The problem was loaded from the JSON file.")
        except Exception as exc:
            messagebox.showerror("Incarcare esuata", f"Fisierul JSON nu a putut fi incarcat.\n\nDetalii: {exc}")


    def load_example(self):
        self.n_var.set(2)
        self.m_constr.set(3)
        self.generate_table()

        c = [3, 5]
        A = [[1, 0], [0, 2], [3, 2]]
        b = [4, 12, 18]
        signs = ["<=", "<=", "<="]

        for j, value in enumerate(c):
            self.c_entries[j].insert(0, str(value))
        for i in range(3):
            for j in range(2):
                self.a_entries[i][j].insert(0, str(A[i][j]))
            self.b_entries[i].insert(0, str(b[i]))
            self.sign_boxes[i].set(signs[i])

        self.clear_text_widget(self.log_text)
        self.append_log("Example loaded: max f(x) = 3x1 + 5x2")
        self.append_log("Expected result: x* = (2, 6), f(x*) = 36")

    def _ensure_generator(self):
        c, A, b, signs, signature = self.collect_data()
        if self.generator is None or self.current_data_signature != signature:
            self.reset_run_state(clear_inputs=False, preserve_log=False)
            self.current_data_signature = signature
            self.solver = UniversalSolverLU(objective=self.problem_type.get())
            self.generator = self.solver.iteration_generator(c, A, b, signs)
            self.append_log("The revised simplex generator was initialized according to the CERC notation.")
            self.append_log("Active solver: UniversalSolverLU (dynamic Big M + manual LU + Charnes rule).")
        return c, A, b, signs


    def open_ai_assistant(self):
        AIAssistantWindow(self)

    # ===================== Actiuni solver =====================
    def next_step(self):
        try:
            self._ensure_generator()
            if self.finished:
                messagebox.showinfo("Algorithm completed", "The solve is already complete.")
                return

            state = next(self.generator)
            self.current_state = state
            self._consume_state(state)
        except StopIteration:
            self.finished = True
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.append_log(f"ERROR: {exc}")

    def solve_complete(self):
        try:
            self._ensure_generator()
            if self.finished:
                return
            while True:
                state = next(self.generator)
                self.current_state = state
                self._consume_state(state)
                if state["status"] in {"optimal", "infeasible", "unbounded"}:
                    break
        except StopIteration:
            self.finished = True
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.append_log(f"ERROR: {exc}")

    def _consume_state(self, state):
        self.finished = state["status"] in {"optimal", "infeasible", "unbounded"}
        self._update_iteration_views(state)

        entering = state.get("entering")
        leaving_row = state.get("leaving_row")
        self._apply_input_highlights(entering=entering, leaving_row=leaving_row)

        self._write_state_log(state)
        if state.get("final_result") is not None:
            self.final_result = state["final_result"]
            self._display_final_result(self.final_result)
            if state["status"] == "infeasible":
                messagebox.showwarning("Incompatible system", "The constraint system is infeasible.")
            elif state["status"] == "unbounded":
                messagebox.showwarning("Unbounded problem", state["message"])
        else:
            self._update_plot(state.get("z_history", []))

    # ===================== Actualizari UI =====================
    def _clear_tree(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    def _update_iteration_views(self, state):
        problem = state["problem"]
        var_names = problem["var_names"]
        basis_names = [var_names[idx] for idx in state["basis"]]

        entering = state.get("entering")
        leaving_var = state.get("leaving_var")
        theta = state.get("theta")

        if entering is None:
            pivot_text = "The optimality condition Δ_j ≤ 0 is satisfied for all nonbasic variables."
        elif leaving_var is None or theta is None:
            pivot_text = (
                f"Entering variable x_k = {var_names[entering]} "
                f"(Δ_k={state['reduced'][entering]:.4f}) | "
                "No leaving variable exists, so the problem is unbounded in this direction."
            )
        else:
            pivot_text = (
                f"Entering variable x_k = {var_names[entering]} (Δ_k={state['reduced'][entering]:.4f}) | "
                f"Leaving variable from B: {var_names[leaving_var]} | "
                f"θ_p = {theta:.6f}"
            )
            if state.get("lexicographic_used"):
                pivot_text += " | Charnes lexicographic rule applied"

        self.iteration_summary.configure(
            text=(
                f"Iteration {state['it']} | Status: {state['status']} | "
                f"Partial f(x*) = {state['z']:.6f} | Z_ext = {state.get('z_ext', state['z']):.6f} | "
                f"M = {problem.get('big_m', 0.0):.6f} | Basis B: {basis_names}\n"
                f"Convention: Δ_j = c_j - z_j, with z_j = π^T P_j | {pivot_text}"
            )
        )

        preview_lines = [
            f"Iteration: {state['it']}",
            f"Status: {state['status']}",
            f"Dynamic M: {problem.get('big_m', 0.0):.6f}",
            f"Basis B: {basis_names}",
            f"x_B = {np.round(state['xb'], 6).tolist()}",
            f"c_B = {np.round(state['cb'], 6).tolist()}",
            f"π = {np.round(state['pi'], 6).tolist()}",
            f"f(x) real: {state['z']:.6f}",
            f"Z_ext = c_B^T x_B: {state.get('z_ext', state['z']):.6f}",
        ]
        if entering is not None:
            preview_lines.append(f"Entering variable x_k: {var_names[entering]} (Δ_k={state['reduced'][entering]:.6f})")
            if leaving_var is None:
                preview_lines.append("Leaving variable from B: none (unbounded direction)")
            else:
                preview_lines.append(f"Leaving variable from B: {var_names[leaving_var]}")
        self.preview_label.configure(text=f"Last state: iteration {state['it']}")
        self.set_text_widget(self.preview_text, "\n".join(preview_lines))

        row_headers = [f"R{i + 1}" for i in range(problem["m"])]
        basis_headers = [var_names[idx] for idx in state["basis"]]
        self.table_B.set_data(state["B"], row_headers=row_headers, col_headers=basis_headers)
        self.table_L.set_data(state["L"], row_headers=row_headers, col_headers=[f"c{i + 1}" for i in range(problem["m"])])
        self.table_U.set_data(state["U"], row_headers=[f"r{i + 1}" for i in range(problem["m"])], col_headers=basis_headers)
        self.table_Aext.set_data(
            problem["A_ext"],
            row_headers=row_headers,
            col_headers=var_names,
            highlight_rows=[state["leaving_row"]] if state["leaving_row"] is not None else [],
            highlight_cols=[state["entering"]] if state["entering"] is not None else [],
        )
        self.table_pi.set_data(np.array(state["pi"]), row_headers=["π"], col_headers=[f"π_{i + 1}" for i in range(problem["m"])])
        self.table_reduced.set_data(np.array(state["reduced"]), row_headers=["Δ_j"], col_headers=var_names)

        if state.get("d") is not None:
            direction = np.vstack([state["d"], state["ratios"] if state["ratios"] is not None else np.full_like(state["d"], np.nan)])
            self.table_direction.set_data(
                direction,
                row_headers=["d = B⁻¹P_k", "θ_i = x_Bi / d_i"],
                col_headers=row_headers,
                highlight_rows=[1] if state.get("leaving_row") is not None else [],
                highlight_cols=[state["leaving_row"]] if state.get("leaving_row") is not None else [],
            )
        else:
            self.table_direction.set_data(None)

        self._update_plot(state.get("z_history", []))
        if getattr(self, "language", "en") != "en":
            self._apply_language()

    def _display_final_result(self, result):
        status_messages = {
            "optimal": f"Solution found: {result['message']}",
            "infeasible": "The constraint system is infeasible.",
            "unbounded": result["message"],
        }
        style_name = "Warn.TLabel" if result["status"] in {"infeasible", "unbounded"} else "Info.TLabel"

        self.z_big_label.configure(text=f"f(x*) = {result['objective_value']:.6f}")
        self.result_status_label.configure(text=status_messages[result["status"]] + " | Notation: f(x*), x_B, Δ = c - z", style=style_name)

        self._clear_tree(self.variables_tree)
        for idx, name in enumerate(result["var_names"]):
            self.variables_tree.insert(
                "",
                "end",
                values=(name, result["var_kinds"][idx], f"{result['x_full'][idx]:.6f}"),
            )

        self._clear_tree(self.constraints_tree)
        for item in result["constraint_statuses"]:
            self.constraints_tree.insert(
                "",
                "end",
                values=(f"R{item['row'] + 1}", f"{item['lhs']:.6f}", item["sign"], f"{item['rhs']:.6f}", item["status"]),
            )

        verify_text = (
            f"Original system feasibility: {'YES' if result['feasible_original'] else 'NO'} | "
            f"Verification A_ext · x_ext = b: {'YES' if result['standardized_ok'] else 'NO'} | "
            f"Dynamic M: {result.get('big_m', 0.0):.6f}"
        )
        self.verify_label.configure(text=verify_text + " | CERC notation: x_B, π, Δ = c - z, Z_ext = c_B^T x_B")

        original_table = np.column_stack([result["Ax"], result["b"]])
        self.table_verify_original.set_data(
            original_table,
            row_headers=[f"R{i + 1}" for i in range(len(result["b"]))],
            col_headers=["A·x", "b"],
        )
        standard_table = np.column_stack([result["Aext_xext"], result["b"]])
        self.table_verify_standard.set_data(
            standard_table,
            row_headers=[f"Eq{i + 1}" for i in range(len(result["b"]))],
            col_headers=["A_ext·x_ext", "b"],
        )
        self._update_plot(result["z_history"])
        if getattr(self, "language", "en") != "en":
            self._apply_language()

    def _update_plot(self, z_history):
        self.ax.clear()
        self.ax.set_title(self.localize_text("Monotone convergence of Z_ext = c_B^T x_B"))
        self.ax.set_xlabel(self.localize_text("Iteration"))
        self.ax.set_ylabel(self.localize_text("Z_ext = c_B^T x_B"))
        if z_history:
            x = list(range(1, len(z_history) + 1))
            self.ax.plot(x, z_history, marker="o")
            self.ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas_plot.draw_idle()

    def _write_state_log(self, state):
        problem = state["problem"]
        var_names = problem["var_names"]
        basis_names = [var_names[idx] for idx in state["basis"]]
        self.append_log(f"\n--- Iteration {state['it']} ---")
        self.append_log(f"The dynamic numerical barrier was set to M = {problem.get('big_m', 0.0):.6f}")
        self.append_log(f"Current basis: {basis_names}")
        self.append_log("PLU factorization computed for P * B = L * U.")
        self.append_log(f"π = {np.round(state['pi'], 6).tolist()}")
        self.append_log(f"Δ = c - z = {np.round(state['reduced'], 6).tolist()}")
        self.append_log(f"Original objective value (without penalties) = {state['z']:.6f}")
        self.append_log(f"Z_ext (with M penalties, used for the chart) = {state.get('z_ext', state['z']):.6f}")
        entering = state.get("entering")
        leaving_var = state.get("leaving_var")
        theta = state.get("theta")
        if entering is not None:
            if state.get("lexicographic_used"):
                cand_names = [f"R{idx + 1}" for idx in state.get("lexicographic_candidates", [])]
                self.append_log("Degeneracy detected. Applying the Charnes tie-breaking rule.")
                self.append_log(f"Leaving candidates: {cand_names}")
            if leaving_var is None or theta is None:
                self.append_log(
                    f"Entering {var_names[entering]} | No leaving variable | unbounded direction"
                )
            else:
                self.append_log(
                    f"Entering {var_names[entering]} | Leaving {var_names[leaving_var]} | theta = {theta:.6f}"
                )
        self.append_log(f"Iteration status: {state['status']}")
        if state.get("final_result") is not None:
            self.append_log(f"Final result: {state['final_result']['message']}")



class AIAssistantWindow:
    """
    Chat assistant powered by a local open-source LLM through Ollama.
    The assistant converts natural language optimization problems into JSON
    compatible with the simplex application and can explain the final solution.
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"
    MODEL_NAME = "llama3"

    def __init__(self, app):
        self.app = app
        self.generated_json = None
        self.generated_problem = None
        self.last_raw_response = ""
        self.last_user_prompt = getattr(app, "ai_problem_context", "")

        self.window = tk.Toplevel(app.root)
        self.window.title("AI Optimization Assistant")
        self.window.geometry("950x720")
        self.window.configure(bg="#eff6ff")
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        if hasattr(self.app, "ai_windows"):
            self.app.ai_windows.append(self)

        title = tk.Label(
            self.window,
            text="AI Optimization Assistant",
            bg="#1e3a8a",
            fg="white",
            padx=12,
            pady=10,
            font=("Segoe UI", 13, "bold")
        )
        title.pack(fill="x")

        self.chat = ScrolledText(
            self.window,
            wrap="word",
            font=("Consolas", 10),
            bg="#ffffff",
            fg="#0f172a"
        )
        self.chat.pack(fill="both", expand=True, padx=10, pady=10)

        info = (
            "Explain your optimization problem in natural language.\n"
            "Example: A company produces phones and laptops with labor and material constraints.\n"
            "The AI will generate a JSON problem directly compatible with the simplex solver."
        )
        self.chat.insert(tk.END, self._ui(info) + "\n\n")
        self.chat.configure(state="normal")

        bottom = ttk.Frame(self.window)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        self.entry = tk.Text(
            bottom,
            height=5,
            font=("Segoe UI", 10)
        )
        self.entry.pack(side="left", fill="x", expand=True)

        buttons = ttk.Frame(bottom)
        buttons.pack(side="left", padx=(8, 0))

        ttk.Button(
            buttons,
            text="Send",
            command=self.send_message,
            style="Accent.TButton"
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(
            buttons,
            text="Import JSON",
            command=self.import_json,
            style="Json.TButton"
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(
            buttons,
            text="Explain solution",
            command=self.explain_current_solution,
            style="Soft.TButton"
        ).pack(fill="x", pady=(0, 6))

        ttk.Button(
            buttons,
            text="Clear Chat",
            command=self.clear_chat,
            style="Soft.TButton"
        ).pack(fill="x")

        if getattr(self.app, "language", "en") != "en":
            self.apply_language()

        if getattr(self.app, "final_result", None) is not None:
            self.window.after(300, self.explain_current_solution)

    def _ui(self, text):
        if hasattr(self.app, "localize_text"):
            return self.app.localize_text(text)
        return text

    def _on_close(self):
        try:
            if self in self.app.ai_windows:
                self.app.ai_windows.remove(self)
        except Exception:
            pass
        self.window.destroy()

    def apply_language(self):
        try:
            self.window.title(self._ui("AI Optimization Assistant"))
            self.app._apply_language_to_widget_tree(self.window)
            self.app._translate_text_widget_content(self.chat)
        except Exception:
            pass

    def _append_chat(self, text):
        self.chat.configure(state="normal")
        self.chat.insert(tk.END, text)
        self.chat.see(tk.END)
        self.chat.configure(state="normal")

    def _append_chat_safe(self, text):
        try:
            self.window.after(0, lambda: self._append_chat(text))
        except Exception:
            pass

    def clear_chat(self):
        self.chat.delete("1.0", tk.END)
        self.generated_json = None
        self.generated_problem = None
        info = (
            "Explain your optimization problem in natural language.\n"
            "Example: A company produces phones and laptops with labor and material constraints.\n"
            "The AI will generate a JSON problem directly compatible with the simplex solver."
        )
        self.chat.insert(tk.END, self._ui(info) + "\n\n")

    def send_message(self):
        prompt = self.entry.get("1.0", tk.END).strip()
        if not prompt:
            return

        self.last_user_prompt = prompt
        self._append_chat(f"{self._ui('USER:')}\n{prompt}\n\n")
        self.entry.delete("1.0", tk.END)

        threading.Thread(
            target=self.ask_llm,
            args=(prompt,),
            daemon=True
        ).start()

    def _call_ollama(self, prompt, json_mode=False, timeout=240):
        payload = {
            "model": self.MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "top_p": 0.15,
                "num_ctx": 8192,
                "num_predict": 4096,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(self.OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
        except requests.HTTPError:
            if json_mode:
                payload.pop("format", None)
                response = requests.post(self.OLLAMA_URL, json=payload, timeout=timeout)
                response.raise_for_status()
            else:
                raise

        data = response.json()
        result = str(data.get("response", "")).strip()
        if not result:
            raise ValueError("Ollama returned an empty response.")
        return result

    def _build_problem_prompt(self, user_problem):
        return """
You are an expert in operations research and linear programming.

The user explains a real-world optimization problem. Convert it into a linear programming JSON object for a revised simplex solver.

Return STRICT VALID JSON ONLY. Do not use markdown, explanations, comments, ellipses, or extra text.

The JSON MUST follow EXACTLY this structure:
{
  "problem_type": "max",
  "n": 2,
  "m": 3,
  "c": [3, 5],
  "A": [[1, 0], [0, 2], [3, 2]],
  "b": [4, 12, 18],
  "signs": ["<=", "<=", "<="]
}

Rules:
- Include every key: problem_type, n, m, c, A, b, signs.
- problem_type must be exactly "max" or "min".
- c must have exactly n numeric values.
- A must have exactly m rows and every row must have exactly n numeric values.
- b must have exactly m numeric values.
- signs must have exactly m values and each must be one of: "<=", ">=", "=".
- Keep zero coefficients in c and A. Do not omit them.
- Use <= for "at most", "not more than", "maximum" constraints.
- Use >= for "at least", "minimum" constraints.
- Use = only for exact equality constraints.
- Make the JSON complete and parseable by json.loads.
""".strip() + "\n\nUSER PROBLEM:\n" + user_problem

    def _build_repair_prompt(self, raw_response, error_text):
        return f"""
You must repair the following model output into one complete, strict JSON object for a simplex solver.

Required structure:
{{
  "problem_type": "max" or "min",
  "n": integer,
  "m": integer,
  "c": [numbers, length n],
  "A": [[numbers, length n], ... exactly m rows],
  "b": [numbers, length m],
  "signs": ["<=" or ">=" or "=", length m]
}}

Return ONLY the corrected JSON. Do not explain anything.

Parsing/validation error:
{error_text}

Model output to repair:
{raw_response}
""".strip()

    def _cleanup_json_text(self, text):
        cleaned = str(text).strip()
        cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
        return cleaned

    def _extract_balanced_json_object(self, text):
        cleaned = self._cleanup_json_text(text)
        if not cleaned:
            raise ValueError("The AI response is empty.")

        starts = [idx for idx, char in enumerate(cleaned) if char == "{"]
        if not starts:
            raise ValueError("No JSON object was found in the AI response.")

        for start in starts:
            depth = 0
            in_string = False
            escape = False
            for pos in range(start, len(cleaned)):
                char = cleaned[pos]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return cleaned[start:pos + 1]

        raise ValueError("The JSON object is incomplete or has unbalanced braces.")

    def _clean_number(self, value, label):
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
        number = float(value)
        if abs(number - round(number)) <= 1e-12:
            return int(round(number))
        return number

    def _normalize_sign(self, value):
        sign = str(value).strip().lower().replace(" ", "")
        sign = sign.replace("≤", "<=").replace("≥", ">=")
        if sign in {"<=", "=<", "<", "le", "lessequal", "lessorequal", "atmost", "max"}:
            return "<="
        if sign in {">=", "=>", ">", "ge", "greaterequal", "greaterorequal", "atleast", "min"}:
            return ">="
        if sign in {"=", "==", "eq", "equal", "equals"}:
            return "="
        raise ValueError(f"Invalid constraint sign: {value}")

    def _normalize_problem_data(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("The JSON root must be an object.")
        if isinstance(data.get("problem"), dict):
            data = data["problem"]

        raw_type = data.get("problem_type", data.get("type", data.get("objective_type", data.get("objective", "max"))))
        if isinstance(raw_type, dict):
            raw_type = raw_type.get("sense", "max")
        raw_type = str(raw_type).strip().lower()
        if raw_type in {"maximize", "maximization", "maximum"}:
            raw_type = "max"
        if raw_type in {"minimize", "minimization", "minimum"}:
            raw_type = "min"
        if raw_type not in {"max", "min"}:
            raise ValueError("problem_type must be max or min.")

        if "c" not in data:
            raise ValueError("Missing vector c.")
        if "A" not in data and "a" not in data:
            raise ValueError("Missing matrix A.")
        if "b" not in data:
            raise ValueError("Missing vector b.")
        if "signs" not in data:
            raise ValueError("Missing vector signs.")

        c = [self._clean_number(value, f"c[{idx}]") for idx, value in enumerate(data["c"])]
        A_source = data["A"] if "A" in data else data["a"]
        A = [
            [self._clean_number(value, f"A[{i},{j}]") for j, value in enumerate(row)]
            for i, row in enumerate(A_source)
        ]
        b = [self._clean_number(value, f"b[{idx}]") for idx, value in enumerate(data["b"])]
        signs = [self._normalize_sign(value) for value in data["signs"]]

        n = int(data.get("n", len(c)))
        m = int(data.get("m", len(A)))

        if n <= 0 or m <= 0:
            raise ValueError("n and m must be positive integers.")
        if len(c) != n:
            raise ValueError("The size of vector c does not match n.")
        if len(A) != m:
            raise ValueError("The number of rows in A does not match m.")
        if any(len(row) != n for row in A):
            raise ValueError("Every row in A must have exactly n values.")
        if len(b) != m:
            raise ValueError("The size of vector b does not match m.")
        if len(signs) != m:
            raise ValueError("The size of signs does not match m.")

        return {
            "problem_type": raw_type,
            "n": n,
            "m": m,
            "c": c,
            "A": A,
            "b": b,
            "signs": signs,
        }

    def _parse_problem_json(self, raw_text):
        cleaned = self._cleanup_json_text(raw_text)
        try:
            data = json.loads(cleaned)
        except Exception:
            candidate = self._extract_balanced_json_object(cleaned)
            data = json.loads(candidate)
        return self._normalize_problem_data(data)

    def _generate_problem_with_repair(self, user_problem):
        raw = self._call_ollama(self._build_problem_prompt(user_problem), json_mode=True, timeout=240)
        last_error = None

        for attempt in range(3):
            try:
                return self._parse_problem_json(raw), raw
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                raw = self._call_ollama(
                    self._build_repair_prompt(raw, str(exc)),
                    json_mode=True,
                    timeout=240,
                )

        raise ValueError(f"{self._ui('The AI did not return a complete valid JSON problem. Last error:')} {last_error}")

    def _set_generated_problem(self, problem, raw_response):
        self.generated_problem = problem
        self.generated_json = json.dumps(problem, ensure_ascii=False, indent=2)
        self.last_raw_response = raw_response
        self.app.ai_problem_context = self.last_user_prompt
        self._append_chat(
            f"AI:\n{self._ui('AI generated and validated this JSON problem. You can import it now.')}\n"
            f"{self.generated_json}\n\n"
        )

    def ask_llm(self, prompt):
        try:
            problem, raw_response = self._generate_problem_with_repair(prompt)
            self.window.after(0, lambda: self._set_generated_problem(problem, raw_response))
        except Exception as exc:
            self._append_chat_safe(
                f"{self._ui('AI ERROR:')}\n"
                f"{self._ui('Could not generate a complete JSON problem.')}\n\n"
                f"{self._ui('Make sure Ollama is installed and running.')}\n"
                f"{self._ui('Run in terminal:')}\n"
                f"ollama run {self.MODEL_NAME}\n\n"
                f"{self._ui('Technical details:')} {exc}\n\n"
            )

    def import_json(self):
        if self.generated_problem is not None:
            data = self.generated_problem
        elif self.generated_json:
            try:
                data = self._parse_problem_json(self.generated_json)
            except Exception as exc:
                messagebox.showerror(
                    self._ui("Import failed"),
                    f"{self._ui('The generated JSON could not be imported.')}\n\n{exc}"
                )
                return
        else:
            messagebox.showwarning(
                self._ui("No JSON"),
                self._ui("The AI has not generated a JSON problem yet.")
            )
            return

        try:
            self.app.problem_type.set(data["problem_type"])
            self.app.n_var.set(data["n"])
            self.app.m_constr.set(data["m"])
            self.app.generate_table()

            for j, value in enumerate(data["c"]):
                self.app.c_entries[j].insert(0, str(value))

            for i in range(data["m"]):
                for j in range(data["n"]):
                    self.app.a_entries[i][j].insert(0, str(data["A"][i][j]))
                self.app.b_entries[i].insert(0, str(data["b"][i]))
                self.app.sign_boxes[i].set(data["signs"][i])

            self.app.append_log(self._ui("Optimization problem imported from AI Assistant."))
            self.app.ai_problem_context = self.last_user_prompt

            messagebox.showinfo(
                self._ui("Import successful"),
                self._ui("The AI-generated optimization problem was loaded.")
            )

        except Exception as exc:
            messagebox.showerror(
                self._ui("Import failed"),
                f"{self._ui('The generated JSON could not be imported.')}\n\n{exc}"
            )

    def _plain_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, dict):
            return {key: self._plain_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._plain_value(item) for item in value]
        return value

    def _solution_context(self):
        result = self.app.final_result
        decision_values = {
            f"x{idx + 1}": float(value)
            for idx, value in enumerate(result["x_decision"])
        }
        basis_names = [result["var_names"][idx] for idx in result["basis"]]
        return self._plain_value({
            "original_user_problem_text": getattr(self.app, "ai_problem_context", ""),
            "problem_type": result["objective"],
            "objective_coefficients_c": result["c"],
            "constraint_matrix_A": result["A"],
            "right_hand_side_b": result["b"],
            "constraint_signs": result["signs"],
            "status": result["status"],
            "message": result["message"],
            "decision_variable_solution": decision_values,
            "objective_value": result["objective_value"],
            "basis_at_finish": basis_names,
            "constraint_checks": result["constraint_statuses"],
            "iterations": result["iterations"],
            "feasible_original": result["feasible_original"],
            "standardized_ok": result["standardized_ok"],
        })

    def _build_solution_explanation_prompt(self):
        is_ro = getattr(self.app, "language", "en") == "ro"
        language_name = "Romanian" if is_ro else "English"
        language_rule = (
            "Write the entire answer in Romanian. Do not use English section titles or English explanatory text."
            if is_ro
            else "Write the entire answer in English."
        )
        context = json.dumps(self._solution_context(), ensure_ascii=False, indent=2)
        return f"""
You are a friendly operations research tutor.

Explain the simplex result in {language_name}, simply and contextually. Do not return JSON.
{language_rule}

What to include:
1. State whether the model is optimal, infeasible, or unbounded.
2. Explain the optimal decision variable values in plain words.
3. Explain the objective value mathematically.
4. Mention which constraints are binding and which still have slack/surplus.
5. Add a short intuitive explanation of why this is the final simplex solution.

Avoid heavy jargon. Keep the explanation clear and useful for a student.

Solver context:
{context}
""".strip()

    def _fallback_solution_explanation(self):
        result = self.app.final_result
        status = result["status"]
        is_ro = getattr(self.app, "language", "en") == "ro"

        if status == "infeasible":
            if is_ro:
                return (
                    "Sistemul de restricții este infezabil: nu există valori ale variabilelor de decizie "
                    "care să respecte simultan toate restricțiile. Metoda Big-M a păstrat o variabilă artificială "
                    "pozitivă la final, ceea ce indică imposibilitatea sistemului original."
                )
            return (
                "The constraint system is infeasible: there are no decision-variable values that satisfy "
                "all constraints at the same time. The Big-M method kept a positive artificial variable "
                "at the end, which indicates that the original system is impossible."
            )

        if status == "unbounded":
            if is_ro:
                return (
                    "Problema este nemărginită: există o direcție în care funcția obiectiv poate fi îmbunătățită "
                    "fără să se încalce restricțiile, deci nu există o soluție optimă finită."
                )
            return (
                "The problem is unbounded: there is a direction in which the objective function can keep improving "
                "without violating the constraints, so there is no finite optimal solution."
            )

        decision_text = ", ".join(
            f"x{idx + 1} = {float(value):.6g}"
            for idx, value in enumerate(result["x_decision"])
        )
        binding = [f"R{item['row'] + 1}" for item in result["constraint_statuses"] if item["status"] == "binding"]
        nonbinding = [f"R{item['row'] + 1}" for item in result["constraint_statuses"] if item["status"] == "nonbinding"]

        if is_ro:
            binding_text = ", ".join(binding) if binding else "niciuna"
            nonbinding_text = ", ".join(nonbinding) if nonbinding else "niciuna"
            return (
                f"Soluția optimă găsită este {decision_text}. Valoarea funcției obiectiv este "
                f"f(x*) = {float(result['objective_value']):.6g}. Restricțiile active, adică cele folosite exact "
                f"la limită, sunt: {binding_text}. Restricțiile neactive, care mai au slack/surplus, sunt: "
                f"{nonbinding_text}. Matematic, simplexul s-a oprit deoarece nu mai există o variabilă nebazică "
                "ce poate îmbunătăți funcția obiectiv conform costurilor reduse, deci baza curentă este optimă."
            )

        binding_text = ", ".join(binding) if binding else "none"
        nonbinding_text = ", ".join(nonbinding) if nonbinding else "none"
        return (
            f"The optimal solution is {decision_text}. The objective value is "
            f"f(x*) = {float(result['objective_value']):.6g}. The binding constraints, meaning the ones used exactly "
            f"at their limits, are: {binding_text}. The nonbinding constraints, which still have slack/surplus, are: "
            f"{nonbinding_text}. Mathematically, the simplex method stopped because there is no nonbasic variable "
            "that can improve the objective according to the reduced costs, so the current basis is optimal."
        )

    def explain_current_solution(self):
        if getattr(self.app, "final_result", None) is None:
            messagebox.showwarning(
                self._ui("No solution"),
                self._ui("Solve the problem first, then AI can explain the final mathematical solution.")
            )
            return

        self._append_chat(f"AI:\n{self._ui('AI is preparing a simple explanation of the current solution...')}\n\n")
        threading.Thread(
            target=self.ask_solution_explanation,
            daemon=True
        ).start()

    def ask_solution_explanation(self):
        try:
            explanation = self._call_ollama(
                self._build_solution_explanation_prompt(),
                json_mode=False,
                timeout=240,
            ).strip()
            if not explanation:
                raise ValueError(self._ui("Empty explanation."))
            self._append_chat_safe(f"AI:\n{explanation}\n\n")
        except Exception as exc:
            fallback = self._fallback_solution_explanation()
            if getattr(self.app, "language", "en") == "ro":
                note = (
                    "Notă: explicația de mai sus a fost generată local din rezultatele solverului, "
                    f"deoarece Ollama nu a returnat o explicație completă. Detalii: {exc}"
                )
            else:
                note = (
                    "Note: the explanation above was generated locally from the solver results, "
                    f"because Ollama did not return a complete explanation. Details: {exc}"
                )
            self._append_chat_safe(f"AI:\n{fallback}\n\n{note}\n\n")

def main():
    root = tk.Tk()
    app = SimplexInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
