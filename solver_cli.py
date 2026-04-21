import json
import sys
from pathlib import Path

import numpy as np


class UniversalSolverLU:
    """
    Standalone mathematical engine for the revised simplex method.

    Features:
    - supports max / min;
    - supports <=, >=, = constraints;
    - uses Big-M for artificial variables;
    - uses manual PLU factorization for linear solves;
    - exposes the algorithm as a generator, so every step can be logged.
    """

    def __init__(self, objective="max", tol=1e-10, max_iter=100, big_m=1e6):
        self.objective = objective.lower().strip()
        if self.objective not in {"max", "min"}:
            raise ValueError("Objective type must be 'max' or 'min'.")
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.M = float(big_m)

    def plu_decomposition(self, matrix):
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

    def _normalize_rows(self, A, b, signs):
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

        self.M = float((np.sum(np.abs(c)) + 1.0) * 1000.0)

        A_ext = np.array(A, dtype=float, copy=True)
        c_ext = np.array(c, dtype=float, copy=True)
        basis = []
        artificial_indices = []
        var_names = [f"x{j + 1}" for j in range(n_orig)]
        var_kinds = ["x (decision)" for _ in range(n_orig)]

        m_cost = -self.M if self.objective == "max" else self.M

        for i in range(m):
            sign = signs[i]

            if sign == "<=":
                col = np.zeros((m, 1), dtype=float)
                col[i, 0] = 1.0
                A_ext = np.hstack([A_ext, col])
                c_ext = np.append(c_ext, 0.0)
                idx = A_ext.shape[1] - 1
                basis.append(idx)
                var_names.append(f"y{i + 1}")
                var_kinds.append("y (slack)")

            elif sign == ">=":
                s_col = np.zeros((m, 1), dtype=float)
                s_col[i, 0] = -1.0
                A_ext = np.hstack([A_ext, s_col])
                c_ext = np.append(c_ext, 0.0)
                var_names.append(f"y{i + 1}^-")
                var_kinds.append("y (surplus)")

                a_col = np.zeros((m, 1), dtype=float)
                a_col[i, 0] = 1.0
                A_ext = np.hstack([A_ext, a_col])
                c_ext = np.append(c_ext, m_cost)
                a_idx = A_ext.shape[1] - 1
                basis.append(a_idx)
                artificial_indices.append(a_idx)
                var_names.append(f"z{i + 1}")
                var_kinds.append("z (artificial)")

            else:
                a_col = np.zeros((m, 1), dtype=float)
                a_col[i, 0] = 1.0
                A_ext = np.hstack([A_ext, a_col])
                c_ext = np.append(c_ext, m_cost)
                a_idx = A_ext.shape[1] - 1
                basis.append(a_idx)
                artificial_indices.append(a_idx)
                var_names.append(f"z{i + 1}")
                var_kinds.append("z (artificial)")

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
            "m": m,
            "n": n_orig,
            "total_vars": A_ext.shape[1],
            "big_m": float(self.M),
            "m_cost": float(m_cost),
        }

    def build_solution(self, basis, xb, total_vars):
        x = np.zeros(total_vars, dtype=float)
        for i, idx in enumerate(basis):
            x[idx] = xb[i]
        return x

    def _compute_basis_inverse(self, P, L, U):
        m = L.shape[0]
        cols = []
        for k in range(m):
            e_k = np.zeros(m, dtype=float)
            e_k[k] = 1.0
            cols.append(self.solve_plu(P, L, U, e_k))
        return np.column_stack(cols)

    def _lex_compare(self, left, right):
        for a, b in zip(left, right):
            diff = float(a) - float(b)
            if abs(diff) <= max(self.tol, 1e-12):
                continue
            return -1 if diff < 0 else 1
        return 0

    def _select_entering_variable(self, reduced_costs, non_basis):
        if self.objective == "max":
            candidates = [j for j in non_basis if reduced_costs[j] > self.tol]
            entering = max(candidates, key=lambda j: (reduced_costs[j], -j)) if candidates else None
        else:
            candidates = [j for j in non_basis if reduced_costs[j] < -self.tol]
            entering = min(candidates, key=lambda j: (reduced_costs[j], j)) if candidates else None
        return entering, candidates

    def _lexicographic_leaving_row(self, xb, B_inv, d, candidates):
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
            statuses.append(
                {
                    "row": i,
                    "lhs": float(lhs[i]),
                    "rhs": float(b[i]),
                    "sign": sign,
                    "status": status,
                }
            )
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
        }

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

            P, L, U = self.plu_decomposition(B)
            pi = self.solve_transposed_plu(P, L, U, cb)
            reduced_costs = c_ext - (A_ext.T @ pi)

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
                "entering_candidates": list(candidates),
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
                    else "The optimality condition is satisfied for the convention Delta = c - z."
                )
                state["final_result"] = self._build_result(
                    problem, basis, xb, state["status"], state["message"], z_history, it
                )
                yield state
                return

            d = self.solve_plu(P, L, U, A_ext[:, entering])
            state["entering"] = entering
            state["d"] = d.copy()

            if np.all(d <= self.tol):
                state["status"] = "unbounded"
                state["message"] = "The problem is unbounded in the direction of the entering variable."
                state["final_result"] = self._build_result(
                    problem, basis, xb, state["status"], state["message"], z_history, it
                )
                yield state
                return

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

            xb = xb - theta * d
            xb[leaving_row] = theta
            xb[np.abs(xb) < self.tol] = 0.0
            basis[leaving_row] = entering

        raise ValueError(
            f"The maximum number of iterations ({self.max_iter}) was reached without a conclusion."
        )

    def solve_complete(self, c, A, b, signs):
        last_state = None
        for state in self.iteration_generator(c, A, b, signs):
            last_state = state
        if last_state is None:
            raise ValueError("The algorithm produced no iterations.")
        return last_state


# ===================== Console logging =====================
def fmt_array(arr, precision=6):
    a = np.array(arr, dtype=float)
    return np.array2string(a, precision=precision, suppress_small=False)


def print_problem_summary(c, A, b, signs, objective):
    print("=" * 78)
    print("REVISED SIMPLEX SOLVER - CONSOLE LOG")
    print("=" * 78)
    print(f"Objective : {objective}")
    print(f"c         : {fmt_array(c)}")
    print(f"A         :\n{fmt_array(A)}")
    print(f"b         : {fmt_array(b)}")
    print(f"signs     : {list(signs)}")
    print("=" * 78)


def print_iteration_log(state):
    problem = state["problem"]
    names = problem["var_names"]
    basis_names = [names[idx] for idx in state["basis"]]
    entering = state["entering"]
    leaving_var = state["leaving_var"]

    print(f"\n--- Iteration {state['it']} ---")
    print(f"Basis variables     : {basis_names}")
    print(f"x_B                 : {fmt_array(state['xb'])}")
    print(f"c_B                 : {fmt_array(state['cb'])}")
    print(f"Big-M               : {problem['big_m']:.6f}")
    print(f"z (original)        : {state['z']:.6f}")
    print(f"z_ext = c_B^T x_B   : {state['z_ext']:.6f}")
    print(f"pi                  : {fmt_array(state['pi'])}")
    print(f"Reduced costs Delta : {fmt_array(state['reduced'])}")

    if state.get("entering_candidates"):
        cand_names = [names[j] for j in state["entering_candidates"]]
        print(f"Improving candidates: {cand_names}")

    if entering is not None:
        print(f"Entering variable   : {names[entering]} (index {entering})")
        print(f"Direction d         : {fmt_array(state['d'])}")
        print(f"Ratios theta_i      : {fmt_array(state['ratios'])}")
        print(f"Chosen theta        : {state['theta']:.6f}")
        print(f"Leaving row         : {state['leaving_row']}")
        print(f"Leaving variable    : {names[leaving_var]} (index {leaving_var})")

    if state.get("lexicographic_used"):
        rows = [f"R{idx + 1}" for idx in state.get("lexicographic_candidates", [])]
        print(f"Lexicographic tie   : applied on rows {rows}")

    print(f"Status              : {state['status']}")
    print(f"Message             : {state['message']}")


def print_final_result(result):
    print("\n" + "=" * 78)
    print("FINAL RESULT")
    print("=" * 78)
    print(f"Status              : {result['status']}")
    print(f"Message             : {result['message']}")
    print(f"Objective value     : {result['objective_value']:.6f}")
    print(f"Decision variables  : {fmt_array(result['x_decision'])}")
    print(f"Full solution       : {fmt_array(result['x_full'])}")
    print(f"Basis indices       : {result['basis']}")
    print(f"Feasible original   : {result['feasible_original']}")
    print(f"Standardized check  : {result['standardized_ok']}")
    print("Constraint statuses :")
    for item in result["constraint_statuses"]:
        print(
            f"  R{item['row'] + 1}: {item['lhs']:.6f} {item['sign']} {item['rhs']:.6f} -> {item['status']}"
        )
    print(f"Z history           : {fmt_array(result['z_history'])}")
    print("=" * 78)


# ===================== Input helpers =====================
def load_problem_from_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    objective = str(data.get("problem_type", data.get("type", data.get("obj", "max")))).strip().lower()
    c = data["c"]
    A = data["A"] if "A" in data else data["a"]
    b = data["b"]
    signs = data["signs"]

    return objective, c, A, b, signs


def default_example():
    objective = "max"
    c = [3, 5]
    A = [[1, 0], [0, 2], [3, 2]]
    b = [4, 12, 18]
    signs = ["<=", "<=", "<="]
    return objective, c, A, b, signs


def read_int(prompt_text, min_value=None):
    while True:
        raw = input(prompt_text).strip()
        try:
            value = int(raw)
            if min_value is not None and value < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue
            return value
        except ValueError:
            print("Invalid integer. Try again.")


def read_float(prompt_text):
    while True:
        raw = input(prompt_text).strip().replace(",", ".")
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")


def read_sign(prompt_text):
    while True:
        raw = input(prompt_text).strip()
        if raw in {"<=", ">=", "="}:
            return raw
        print("Invalid sign. Allowed values are: <=, >=, =")


def read_choice(prompt_text, allowed):
    allowed_lower = {item.lower() for item in allowed}
    while True:
        raw = input(prompt_text).strip().lower()
        if raw in allowed_lower:
            return raw
        print(f"Invalid choice. Allowed values: {', '.join(allowed)}")


def read_problem_from_keyboard():
    print("Interactive input mode")
    print("Enter the linear programming problem step by step.")
    print()

    objective = read_choice("Objective (max/min): ", ["max", "min"])
    n = read_int("Number of decision variables n: ", min_value=1)
    m = read_int("Number of constraints m: ", min_value=1)

    c = []
    print("\nObjective coefficients:")
    for j in range(n):
        c.append(read_float(f"  c[{j + 1}] = "))

    A = []
    b = []
    signs = []

    print("\nConstraint coefficients:")
    for i in range(m):
        row = []
        print(f"Constraint {i + 1}:")
        for j in range(n):
            row.append(read_float(f"  A[{i + 1},{j + 1}] = "))
        sign = read_sign("  Sign (<=, >=, =): ")
        rhs = read_float(f"  b[{i + 1}] = ")
        A.append(row)
        signs.append(sign)
        b.append(rhs)
        print()

    return objective, c, A, b, signs


def choose_input_mode(argv):
    if len(argv) > 1:
        input_path = Path(argv[1])
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {input_path}")
        return load_problem_from_json(input_path)

    print("Choose input mode:")
    print("  1 - interactive keyboard input")
    print("  2 - built-in example")
    print("  3 - load from JSON path")
    mode = read_choice("Your choice (1/2/3): ", ["1", "2", "3"])

    if mode == "1":
        return read_problem_from_keyboard()
    if mode == "2":
        return default_example()

    json_path = input("JSON file path: ").strip()
    if not json_path:
        raise ValueError("No JSON path provided.")
    return load_problem_from_json(Path(json_path))


# ===================== Main =====================
def main():
    objective, c, A, b, signs = choose_input_mode(sys.argv)

    print_problem_summary(c, A, b, signs, objective)

    solver = UniversalSolverLU(objective=objective)
    final_state = None

    for state in solver.iteration_generator(c, A, b, signs):
        print_iteration_log(state)
        final_state = state

    if final_state is None:
        raise RuntimeError("The solver produced no states.")

    print_final_result(final_state["final_result"])


if __name__ == "__main__":
    main()
