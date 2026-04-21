import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText

import json

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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
        self.root.title("Revised Simplex Pro V9 - CERC Notation / Azure Edition")
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

        self._build_style()
        self._build_layout()
        self.generate_table()

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

        top = ttk.LabelFrame(self.tab_input, text="Problem setup", padding=12)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 10))

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
    def _on_input_configure(self, _event=None):
        self.input_canvas.configure(scrollregion=self.input_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.input_canvas.itemconfigure(self.input_window, width=max(event.width, self.input_inner.winfo_reqwidth()))

    def _on_process_configure(self, _event=None):
        self.process_canvas.configure(scrollregion=self.process_canvas.bbox("all"))

    def _on_process_canvas_configure(self, event):
        self.process_canvas.itemconfigure(self.process_window, width=max(event.width, self.process_inner.winfo_reqwidth()))

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
        ttk.Label(
            self.input_inner,
            text="CERC notation: x for variables, y for slack/surplus variables, z for artificial variables; keyboard navigation and JSON persistence.",
            style="Info.TLabel",
        ).grid(row=1, column=0, columnspan=n + 3, sticky="w", pady=(0, 12))

        ttk.Label(self.input_inner, text="Objective function f(x) =", style="PanelTitle.TLabel").grid(
            row=2, column=0, padx=6, pady=(2, 10), sticky="w"
        )
        for j in range(n):
            cell = ttk.Frame(self.input_inner)
            cell.grid(row=2, column=j + 1, padx=6, pady=(0, 10), sticky="n")
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
            row=3, column=0, columnspan=n + 3, sticky="ew", pady=(6, 12)
        )
        ttk.Label(self.input_inner, text="System Ax = b after standardization", style="PanelTitle.TLabel").grid(
            row=4, column=0, columnspan=n + 3, sticky="w", pady=(0, 6)
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
        row_header.grid(row=5, column=0, padx=2, pady=2, sticky="nsew")

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
            lbl.grid(row=5, column=j + 1, padx=2, pady=2, sticky="nsew")
        tk.Label(
            self.input_inner,
            text="Sign",
            bg="#dbeafe",
            fg="#1e3a8a",
            relief="flat",
            padx=8,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=5, column=n + 1, padx=2, pady=2, sticky="nsew")
        tk.Label(
            self.input_inner,
            text="b",
            bg="#dbeafe",
            fg="#1e3a8a",
            relief="flat",
            padx=8,
            pady=6,
            font=("Segoe UI", 9, "bold"),
        ).grid(row=5, column=n + 2, padx=2, pady=2, sticky="nsew")

        for i in range(m):
            row = 6 + i
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
            row=6 + m, column=0, columnspan=n + 3, sticky="w", pady=(12, 0)
        )

        self._on_input_configure()
        if self.c_entries:
            self.c_entries[0].focus_set()
            self._scroll_widget_into_view(self.c_entries[0])

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
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state="disabled")

    def append_log(self, text):
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

        if state["entering"] is None:
            pivot_text = "The optimality condition Δ_j ≤ 0 is satisfied for all nonbasic variables."
        else:
            pivot_text = (
                f"Entering variable x_k = {var_names[state['entering']]} (Δ_k={state['reduced'][state['entering']]:.4f}) | "
                f"Leaving variable from B: {var_names[state['leaving_var']]} | "
                f"θ_p = {state['theta']:.6f}"
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
        if state.get("entering") is not None:
            preview_lines.append(f"Entering variable x_k: {var_names[state['entering']]} (Δ_k={state['reduced'][state['entering']]:.6f})")
            preview_lines.append(f"Leaving variable from B: {var_names[state['leaving_var']]}")
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

    def _update_plot(self, z_history):
        self.ax.clear()
        self.ax.set_title("Monotone convergence of Z_ext = c_B^T x_B")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Z_ext = c_B^T x_B")
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
        if state.get("entering") is not None:
            if state.get("lexicographic_used"):
                cand_names = [f"R{idx + 1}" for idx in state.get("lexicographic_candidates", [])]
                self.append_log("Degeneracy detected. Applying the Charnes tie-breaking rule.")
                self.append_log(f"Leaving candidates: {cand_names}")
            self.append_log(
                f"Entering {var_names[state['entering']]} | Leaving {var_names[state['leaving_var']]} | theta = {state['theta']:.6f}"
            )
        self.append_log(f"Iteration status: {state['status']}")
        if state.get("final_result") is not None:
            self.append_log(f"Final result: {state['final_result']['message']}")


def main():
    root = tk.Tk()
    app = SimplexInterface(root)
    root.mainloop()


if __name__ == "__main__":
    main()
