"""Nested aggregation tests (Step 2 of IR-2).

The compiler derives a multi-pass plan (intermediate grain -> filter -> terminal
grain) from the grain lattice. The IR-2 the LLM emits contains NO step list, no
group keys, no pass ordering - only dimension names, an entity predicate, and
measure names. These tests assert correct numbers for the five design walkthroughs,
including the ambiguity case the compiler must refuse to guess.
"""
import pandas as pd

from compiler.core import compile_logical
from agents.plan_executor import execute_plan


def _df():
    """cust 111: 4 loans in branch A / region R1 (2 NPA).
       cust 222: 2 loans in A / R1.
       cust 333: 3 loans in B / R1 (3 NPA).
       cust 444: 4 loans in C / R2 (1 NPA).
       Every loan: SOH 100; Cum Coll 90; Cum Due-Inst 50; Cum Due-Exp 50."""
    _names = {111: "Alice", 222: "Bob", 333: "Carol", 444: "Dave"}
    rows = []
    def add(loan, cust, unit, region, bucket):
        rows.append({
            "Loan No": loan, "Cust Mob No": cust, "Cust Name": _names[cust],
            "Unit": unit, "RegionName": region,
            "curr_bucket": bucket, "SOH": 100.0,
            "Cum Coll (Inst+Exp)": 90.0, "Cum Due-Inst": 50.0, "Cum Due-Exp": 50.0,
        })
    for i in range(1, 5):  add(i,   111, "A", "R1", "NPA" if i <= 2 else "STD")
    for i in range(5, 7):  add(i,   222, "A", "R1", "STD")
    for i in range(7, 10): add(i,   333, "B", "R1", "NPA")
    for i in range(10, 14):add(i,   444, "C", "R2", "NPA" if i == 10 else "STD")
    return pd.DataFrame(rows)


def _run(ir):
    df = _df()
    plan, errs = compile_logical(ir, df.columns)
    assert errs == [], errs
    out, err = execute_plan(df, plan)
    assert err == ""
    return out


class TestQuery1_CustomersPerBranchWithManyLoans:
    def test(self):
        out = _run({
            "dimensions": ["branch"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "count", "distinct": "Loan No", "op": ">", "value": 3}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "customer_count"}],
        })
        got = dict(zip(out["Unit"], out["customer_count"]))
        assert got.get("A") == 1            # cust 111 (4 loans)
        assert got.get("C") == 1            # cust 444 (4 loans)
        assert "B" not in got               # cust 333 has 3 (not > 3) -> branch drops out


class TestQuery2_FleetOperatorsWithExposure:
    def test(self):
        out = _run({
            "dimensions": ["region"],
            "entity_filters": [{"concept": "fleet_operator"}],   # >= 3 loans, from registry
            "measures": [
                {"agg": "count", "distinct": "Cust Mob No", "alias": "fleet_operators"},
                {"metric": "exposure", "alias": "total_exposure"},
            ],
        })
        r1 = out[out["RegionName"] == "R1"].iloc[0]
        r2 = out[out["RegionName"] == "R2"].iloc[0]
        assert r1["fleet_operators"] == 2 and r1["total_exposure"] == 700.0  # 111(400)+333(300)
        assert r2["fleet_operators"] == 1 and r2["total_exposure"] == 400.0  # 444


class TestQuery3_BorrowersWithMultipleNpaLoans:
    def test(self):
        out = _run({
            "dimensions": ["branch"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "count", "concept": "npa", "op": ">=", "value": 2}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "borrowers"}],
        })
        got = dict(zip(out["Unit"], out["borrowers"]))
        assert got.get("A") == 1            # cust 111 (2 NPA)
        assert got.get("B") == 1            # cust 333 (3 NPA)
        assert "C" not in got               # cust 444 has only 1 NPA


class TestQuery4_FleetLccRatioRollup:
    def test(self):
        out = _run({
            "dimensions": ["region"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"metric": "lcc_pct", "alias": "fleet_lcc"}],
        })
        # All fleet loans: 90/100 -> 90% (sum(num)/sum(den)*100), capped at 100.
        assert out[out["RegionName"] == "R1"].iloc[0]["fleet_lcc"] == 90.0
        assert out[out["RegionName"] == "R2"].iloc[0]["fleet_lcc"] == 90.0


class TestQuery5_AmbiguityIsRefused:
    def test_multi_branch_per_region_is_grain_ambiguous(self):
        # Predicate aggregates over Unit (branch) but output is region: a customer
        # can span branches across regions -> compiler must refuse, not guess.
        plan, errs = compile_logical({
            "dimensions": ["region"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "nunique", "column": "Unit", "op": ">=", "value": 3}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "multi_branch"}],
        }, _df().columns)
        assert any("grain ambiguity" in e for e in errs)


class TestStructureIsCompilerDerived:
    def test_ir_has_no_step_list(self):
        # The IR-2 the LLM emits declares intent only; the multi-pass STRUCTURE is
        # produced by the compiler. Two group_aggregate passes appear in the PLAN,
        # never in the IR.
        ir = {
            "dimensions": ["branch"],
            "entity_filters": [{"entity": "customer",
                                "having": [{"agg": "count", "distinct": "Loan No", "op": ">", "value": 3}]}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "customer_count"}],
        }
        assert "op" not in ir and "plan" not in ir and "stages" not in ir
        plan, errs = compile_logical(ir, _df().columns)
        assert errs == []
        passes = [s for s in plan if s["op"] == "group_aggregate"]
        assert len(passes) == 2                       # intermediate + terminal, derived
        assert passes[0]["group_by"] == ["Unit", "Cust Mob No"]
        assert passes[1]["group_by"] == ["Unit"]


class TestNonNestableMeasureRejected:
    def test_mean_cannot_roll_up(self):
        plan, errs = compile_logical({
            "dimensions": ["branch"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"column": "SOH", "agg": "mean", "alias": "avg_soh"}],
        }, _df().columns)
        assert any("cannot be rolled up" in e for e in errs)


class TestQuery6_DimensionMatchesEntityKey:
    """"Give me all fleet owners in NPA" -> dimensions:["customer"] +
    entity_filters:[fleet_operator], both keyed on the same column
    (Cust Mob No). Regression for a real production bug: an undeduped
    group_by+entity_key produced a duplicate-named group_by list, and
    pandas' groupby(...).size().reset_index() raised
    "cannot insert Cust Mob No, already exists"."""

    def test_customer_dimension_same_as_entity_key_does_not_crash(self):
        out = _run({
            "dimensions": ["customer"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [],
        })
        assert set(out["Cust Mob No"]) == {111, 333, 444}   # each has >= FLEET_MIN_LOANS (3) loans

    def test_intermediate_group_by_has_no_duplicate_columns(self):
        plan, errs = compile_logical({
            "dimensions": ["customer"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [],
        }, _df().columns)
        assert errs == []
        inter = plan[0]
        assert inter["op"] == "group_aggregate"
        assert inter["group_by"] == ["Cust Mob No"]

    def test_cust_name_attached_for_display(self):
        # Result must be readable (a name), not just the raw mobile number the
        # customer grain is keyed on.
        out = _run({
            "dimensions": ["customer"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [],
        })
        got = dict(zip(out["Cust Mob No"], out["Cust Name"]))
        assert got == {111: "Alice", 333: "Carol", 444: "Dave"}

    def test_branch_dimension_unaffected_by_cust_name_attach(self):
        # Cust Mob No only appears as an INTERMEDIATE key here (dimension is
        # branch) -- must not spuriously attach Cust Name to a branch-grain row.
        plan, errs = compile_logical({
            "dimensions": ["branch"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"agg": "count", "distinct": "Cust Mob No", "alias": "customer_count"}],
        }, _df().columns)
        assert errs == []
        term_aggs = plan[2]["aggregations"]
        assert "Cust Name" not in [a["alias"] for a in term_aggs]


class TestQuery7_SingleValueCustomerNameAttached:
    def test_single_pass_customer_grain_attaches_name(self):
        # No entity_filters -> SINGLE-PASS path, not _build_nested.
        out = _run({
            "dimensions": ["customer"],
            "measures": [{"agg": "sum", "column": "SOH", "alias": "total_soh"}],
        })
        got = dict(zip(out["Cust Mob No"], out["Cust Name"]))
        assert got[111] == "Alice"
        assert got[222] == "Bob"


class TestQuery8_EntityFilterOnLoanTableReturnsFullRows:
    """"Give me all loans for fleet owners in NPA, show all columns" -- entity_filters
    with NO dimensions (loan_table intent) must return every original loan-level
    column for the qualifying customers' rows, not collapse to one row per customer."""

    def test_returns_all_original_rows_and_columns_for_qualifying_customers(self):
        plan, errs = compile_logical({
            "intent": "loan_table",
            "entity_filters": [{"concept": "fleet_operator"}],
            "show_all_columns": True,
        }, _df().columns)
        assert errs == [], errs
        df = _df()
        out, err = execute_plan(df, plan)
        assert err == ""
        # cust 111, 333, 444 qualify (>=3 loans); cust 222 (2 loans) is excluded.
        assert set(out["Cust Mob No"]) == {111, 333, 444}
        assert len(out) == 4 + 3 + 4          # every loan row for each qualifying customer
        # ALL original columns survive ("Rank" is execute_plan's own universal addition).
        assert set(out.columns) == set(df.columns) | {"Rank"}

    def test_non_loan_table_intent_does_not_use_the_semi_join_path(self):
        # "How many fleet owners are in NPA" (single_value, no dimensions) must
        # NOT silently return raw per-loan rows via the semi-join path -- that
        # would answer a different question (a loan count) than what was asked
        # (a fleet-owner count). Preserves the pre-existing error instead.
        plan, errs = compile_logical({
            "intent": "single_value",
            "entity_filters": [{"concept": "fleet_operator"}],
        }, _df().columns)
        assert any("nested aggregation currently requires a grouping dimension" in e for e in errs)

    def test_combines_with_a_top_level_filter(self):
        # Top-level filters apply BEFORE the entity predicate is evaluated
        # (matches the existing entity_filters/aggregation convention -- see
        # TestQuery3, where "count >= 2" is checked against an already-NPA-
        # scoped concept): "fleet owner" here means >=3 loans WITHIN the
        # NPA-filtered rows, not >=3 loans overall.
        plan, errs = compile_logical({
            "intent": "loan_table",
            "filters": [{"column": "curr_bucket", "op": "==", "value": "NPA"}],
            "entity_filters": [{"concept": "fleet_operator"}],
            "show_all_columns": True,
        }, _df().columns)
        assert errs == [], errs
        out, err = execute_plan(_df(), plan)
        assert err == ""
        # cust111: 2 NPA loans (< 3, excluded); cust333: 3 NPA loans (qualifies);
        # cust444: 1 NPA loan (excluded).
        assert set(out["Cust Mob No"]) == {333}
        assert len(out) == 3
        assert set(out["curr_bucket"]) == {"NPA"}


class TestQuery9_MisplacedEntityConceptInFilters:
    """Regression for a real production failure: "give me all fleet owners name
    that are in npa" and "...show all columns" both hard-failed with
    "Could not compile query: unknown concept 'fleet_operator'" because the
    Planner put fleet_operator in "filters" instead of "entity_filters".
    fleet_operator only exists in ENTITY_CONCEPTS, never in CONCEPTS, so this
    is unambiguous and the compiler should just route it correctly."""

    def test_entity_concept_in_filters_still_compiles_with_dimensions(self):
        plan, errs = compile_logical({
            "intent": "aggregation",
            "dimensions": ["customer"],
            "filters": [{"concept": "fleet_operator"}, {"concept": "npa"}],
            "entity_filters": [],
            "measures": [],
        }, _df().columns)
        assert errs == [], errs
        out, err = execute_plan(_df(), plan)
        assert err == ""
        assert set(out["Cust Mob No"]) == {333}   # only cust 333 has >=3 NPA loans

    def test_entity_concept_in_filters_still_compiles_for_loan_table(self):
        plan, errs = compile_logical({
            "intent": "loan_table",
            "filters": [{"concept": "fleet_operator"}, {"concept": "npa"}],
            "entity_filters": [],
            "show_all_columns": True,
        }, _df().columns)
        assert errs == [], errs
        out, err = execute_plan(_df(), plan)
        assert err == ""
        assert set(out["Cust Mob No"]) == {333}
        assert len(out) == 3

    def test_entity_concept_as_a_count_measure_is_redundant_not_an_error(self):
        # Real production IR: the Planner ALSO tried to "count" fleet_operator
        # as a measure, on top of already using it correctly in entity_filters.
        # {"agg":"count","concept":"fleet_operator"} would otherwise hit
        # _expand_filters' CONCEPTS-only lookup and fail the same way.
        plan, errs = compile_logical({
            "intent": "aggregation",
            "dimensions": ["customer"],
            "filters": [{"concept": "npa"}],
            "measures": [{"agg": "count", "concept": "fleet_operator", "alias": "fleet_operator_count"}],
            "entity_filters": [{"concept": "fleet_operator"}],
        }, _df().columns)
        assert errs == [], errs
        out, err = execute_plan(_df(), plan)
        assert err == ""
        assert set(out["Cust Mob No"]) == {333}
        # Redundant measure resolves to "how many qualifying entities", i.e. 1 per row.
        assert list(out["fleet_operator_count"]) == [1]

    def test_regular_concept_filters_still_work_unchanged(self):
        # Sanity: a normal CONCEPTS entry (not an entity concept) must still
        # flow through the plain filter path, not get reclassified.
        plan, errs = compile_logical({
            "intent": "loan_table",
            "filters": [{"concept": "npa"}],
        }, _df().columns)
        assert errs == [], errs
        assert plan[0]["op"] == "filter"
        assert plan[0]["conditions"] == [{"column": "curr_bucket", "op": "==", "value": "NPA"}]


class TestEntitySemiJoinCompositeKey:
    """entity_filters accepts a raw {"entity":..., "having":...} predicate
    directly, not just a registered ENTITY_CONCEPTS name -- so a composite-key
    entity (e.g. executive, keyed on ["MNT NAME", "Unit"], the same name
    recurring across branches per registry/semantic_model.py) can reach
    _op_entity_semi_join's multi-column path even though fleet_operator (the
    only current ENTITY_CONCEPTS entry) never does. Previously untested."""

    def _exec_df(self):
        rows = []
        def add(loan, name, unit, n_more=0):
            for i in range(n_more + 1):
                rows.append({"Loan No": f"{loan}-{i}", "MNT NAME": name, "Unit": unit})
        add("R-A", "RAVI", "A", n_more=2)     # (Ravi, A): 3 loans -- qualifies
        add("R-B", "RAVI", "B", n_more=0)     # (Ravi, B): 1 loan -- same NAME, different branch, excluded
        add("S-A", "SUNIL", "A", n_more=1)    # (Sunil, A): 2 loans -- qualifies
        return pd.DataFrame(rows)

    def test_composite_key_distinguishes_same_name_across_branches(self):
        plan, errs = compile_logical({
            "intent": "loan_table",
            "entity_filters": [{"entity": "executive",
                                "having": [{"agg": "count", "op": ">=", "value": 2}]}],
            "show_all_columns": True,
        }, self._exec_df().columns)
        assert errs == [], errs
        out, err = execute_plan(self._exec_df(), plan)
        assert err == ""
        pairs = set(zip(out["MNT NAME"], out["Unit"]))
        assert pairs == {("RAVI", "A"), ("SUNIL", "A")}
        assert len(out) == 3 + 2   # (Ravi,A)'s 3 loans + (Sunil,A)'s 2 -- (Ravi,B)'s 1 excluded

    def test_blank_key_value_is_excluded_not_crashed(self):
        df = self._exec_df()
        df.loc[len(df)] = {"Loan No": "BLANK-0", "MNT NAME": None, "Unit": "A"}
        plan, errs = compile_logical({
            "intent": "loan_table",
            "entity_filters": [{"entity": "executive",
                                "having": [{"agg": "count", "op": ">=", "value": 2}]}],
            "show_all_columns": True,
        }, df.columns)
        assert errs == [], errs
        out, err = execute_plan(df, plan)
        assert err == ""
        # A blank MNT NAME never matches any qualifying (name, branch) pair --
        # excluded, consistent with how a blank Cust Mob No is already excluded
        # elsewhere (analysis/portfolio_intelligence.py's compute_fleet_exposure).
        assert "BLANK-0" not in set(out["Loan No"])


class TestQuery10_LoanCountViaNuniqueLoanNo:
    """Regression for a real production failure: "how many loans does each
    fleet owner have" -> a {"agg":"nunique","distinct":"Loan No"} measure hit
    "count_distinct of 'Loan No' cannot be rolled up across a nested grain" --
    a real limitation for a GENUINE distinct count, but Loan No is the atomic
    row key, so nunique(Loan No) is always just a row count here."""

    def test_nunique_loan_no_measure_compiles_and_matches_plain_count(self):
        out = _run({
            "dimensions": ["customer"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"agg": "nunique", "distinct": "Loan No", "alias": "loan_count"}],
        })
        got = dict(zip(out["Cust Mob No"], out["loan_count"]))
        assert got == {111: 4, 333: 3, 444: 4}

    def test_agg_count_with_distinct_key_also_rewrites(self):
        # The real production shape that first slipped past the "agg==nunique"
        # guard: {"agg":"count","distinct":"Loan No"} -- _infer_kind treats
        # this as count_distinct too (it checks "distinct" in d before "agg"),
        # so the rewrite guard must key off the same signal, not "agg" alone.
        out = _run({
            "dimensions": ["customer"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"agg": "count", "distinct": "Loan No", "alias": "loan_count"}],
        })
        got = dict(zip(out["Cust Mob No"], out["loan_count"]))
        assert got == {111: 4, 333: 3, 444: 4}

    def test_nunique_loan_no_via_column_key_also_rewrites(self):
        # Some IR shapes use {"agg":"nunique","column":...} instead of "distinct".
        plan, errs = compile_logical({
            "dimensions": ["branch"],
            "entity_filters": [{"concept": "fleet_operator"}],
            "measures": [{"agg": "nunique", "column": "Loan No", "alias": "loan_count"}],
        }, _df().columns)
        assert errs == [], errs
        term_aggs = plan[2]["aggregations"]
        funcs = {a["alias"]: a["func"] for a in term_aggs}
        assert funcs["loan_count"] == "sum"   # rewritten to a decomposable count, not nunique
