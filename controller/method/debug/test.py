from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.config import get_db_connection


RESULT_DIR = Path(__file__).resolve().parent / "result"

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for non-benchmark envs.
    tqdm = None


class ProgressBar:
    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self._bar = tqdm(total=self.total, unit="step") if tqdm is not None else None

    def update(self, message: str) -> None:
        if self._bar is not None:
            self._bar.set_description(str(message))
            self._bar.update(1)
            if self._bar.n >= self.total:
                self._bar.close()
            return

        self.current = min(self.total, self.current + 1)
        width = 32
        filled = int(round(width * self.current / self.total))
        bar = "#" * filled + "-" * (width - filled)
        percent = 100.0 * self.current / self.total
        sys.stdout.write(f"\r[{bar}] {self.current}/{self.total} {percent:5.1f}%  {message}")
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
        }
    normalized = [float(value) for value in values]
    return {
        "count": int(len(normalized)),
        "min": float(min(normalized)),
        "p25": _percentile(normalized, 0.25),
        "p50": _percentile(normalized, 0.50),
        "p75": _percentile(normalized, 0.75),
        "p90": _percentile(normalized, 0.90),
        "p95": _percentile(normalized, 0.95),
        "p99": _percentile(normalized, 0.99),
        "max": float(max(normalized)),
        "mean": float(statistics.fmean(normalized)),
        "stdev": float(statistics.pstdev(normalized)) if len(normalized) > 1 else 0.0,
    }


def _fetch_one(cur, query: str, params: list[Any] | None = None) -> Any:
    cur.execute(query, params or [])
    row = cur.fetchone()
    return row[0] if row else None


def _fetch_all_dicts(cur, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur.execute(query, params or [])
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def collect_permission_structure(progress: ProgressBar | None = None) -> dict[str, Any]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            scalar_counts = {
                "tenant_count": int(_fetch_one(cur, "SELECT COUNT(DISTINCT user_id) FROM UserRoles;") or 0),
                "role_count": int(_fetch_one(cur, "SELECT COUNT(DISTINCT role_id) FROM UserRoles;") or 0),
                "permission_role_count": int(_fetch_one(cur, "SELECT COUNT(DISTINCT role_id) FROM PermissionAssignment;") or 0),
                "permission_document_count": int(_fetch_one(cur, "SELECT COUNT(DISTINCT document_id) FROM PermissionAssignment;") or 0),
                "permission_assignment_count": int(_fetch_one(cur, "SELECT COUNT(*) FROM PermissionAssignment;") or 0),
                "user_role_assignment_count": int(_fetch_one(cur, "SELECT COUNT(*) FROM UserRoles;") or 0),
                "document_count": int(_fetch_one(cur, "SELECT COUNT(DISTINCT document_id) FROM documentblocks;") or 0),
                "block_count": int(_fetch_one(cur, "SELECT COUNT(*) FROM documentblocks;") or 0),
            }
            if progress is not None:
                progress.update("loaded basic counts")

            acl_rows = _fetch_all_dicts(
                cur,
                """
                WITH document_tenants AS (
                    SELECT
                        pa.document_id,
                        array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    GROUP BY pa.document_id
                ),
                block_counts AS (
                    SELECT document_id, COUNT(*) AS block_count
                    FROM documentblocks
                    GROUP BY document_id
                ),
                acl_groups AS (
                    SELECT
                        tenant_ids,
                        COUNT(*) AS document_count,
                        COALESCE(SUM(block_counts.block_count), 0) AS vector_count
                    FROM document_tenants
                    LEFT JOIN block_counts ON block_counts.document_id = document_tenants.document_id
                    GROUP BY tenant_ids
                )
                SELECT
                    row_number() OVER (ORDER BY document_count DESC, vector_count DESC, tenant_ids) AS acl_id,
                    tenant_ids,
                    cardinality(tenant_ids) AS acl_size,
                    document_count,
                    vector_count
                FROM acl_groups
                ORDER BY document_count DESC, vector_count DESC, tenant_ids;
                """,
            )
            if progress is not None:
                progress.update("computed ACL pattern statistics")

            tenant_rows = _fetch_all_dicts(
                cur,
                """
                WITH tenant_docs AS (
                    SELECT
                        ur.user_id,
                        COUNT(DISTINCT pa.document_id) AS document_count
                    FROM UserRoles ur
                    JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                    GROUP BY ur.user_id
                ),
                tenant_blocks AS (
                    SELECT
                        ur.user_id,
                        COUNT(DISTINCT db.block_id) AS vector_count
                    FROM UserRoles ur
                    JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                    JOIN documentblocks db ON db.document_id = pa.document_id
                    GROUP BY ur.user_id
                ),
                tenant_roles AS (
                    SELECT
                        user_id,
                        COUNT(DISTINCT role_id) AS role_count
                    FROM UserRoles
                    GROUP BY user_id
                ),
                document_tenants AS (
                    SELECT
                        pa.document_id,
                        array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    GROUP BY pa.document_id
                ),
                tenant_acls AS (
                    SELECT
                        tenant_id AS user_id,
                        COUNT(DISTINCT tenant_ids) AS acl_count
                    FROM document_tenants
                    CROSS JOIN LATERAL unnest(tenant_ids) AS tenant_id
                    GROUP BY tenant_id
                ),
                tenant_cooccurrence AS (
                    SELECT
                        tenant_id AS user_id,
                        SUM(cardinality(tenant_ids) - 1) AS cooccurrence_count,
                        AVG(cardinality(tenant_ids) - 1) AS avg_cooccurrence_per_document
                    FROM document_tenants
                    CROSS JOIN LATERAL unnest(tenant_ids) AS tenant_id
                    GROUP BY tenant_id
                )
                SELECT
                    tr.user_id,
                    COALESCE(tr.role_count, 0) AS role_count,
                    COALESCE(ta.acl_count, 0) AS acl_count,
                    COALESCE(td.document_count, 0) AS document_count,
                    COALESCE(tb.vector_count, 0) AS vector_count,
                    COALESCE(tc.cooccurrence_count, 0) AS cooccurrence_count,
                    COALESCE(tc.avg_cooccurrence_per_document, 0.0) AS avg_cooccurrence_per_document
                FROM tenant_roles tr
                LEFT JOIN tenant_acls ta ON ta.user_id = tr.user_id
                LEFT JOIN tenant_docs td ON td.user_id = tr.user_id
                LEFT JOIN tenant_blocks tb ON tb.user_id = tr.user_id
                LEFT JOIN tenant_cooccurrence tc ON tc.user_id = tr.user_id
                ORDER BY document_count DESC, vector_count DESC, tr.user_id;
                """,
            )
            if progress is not None:
                progress.update("computed tenant statistics")

            role_rows = _fetch_all_dicts(
                cur,
                """
                WITH role_users AS (
                    SELECT role_id, COUNT(DISTINCT user_id) AS tenant_count
                    FROM UserRoles
                    GROUP BY role_id
                ),
                role_docs AS (
                    SELECT role_id, COUNT(DISTINCT document_id) AS document_count
                    FROM PermissionAssignment
                    GROUP BY role_id
                ),
                role_blocks AS (
                    SELECT
                        pa.role_id,
                        COUNT(DISTINCT db.block_id) AS vector_count
                    FROM PermissionAssignment pa
                    JOIN documentblocks db ON db.document_id = pa.document_id
                    GROUP BY pa.role_id
                )
                SELECT
                    COALESCE(ru.role_id, rd.role_id, rb.role_id) AS role_id,
                    COALESCE(ru.tenant_count, 0) AS tenant_count,
                    COALESCE(rd.document_count, 0) AS document_count,
                    COALESCE(rb.vector_count, 0) AS vector_count
                FROM role_users ru
                FULL OUTER JOIN role_docs rd ON rd.role_id = ru.role_id
                FULL OUTER JOIN role_blocks rb ON rb.role_id = COALESCE(ru.role_id, rd.role_id)
                ORDER BY document_count DESC, vector_count DESC, role_id;
                """,
            )
            if progress is not None:
                progress.update("computed role statistics")

            pair_rows = _fetch_all_dicts(
                cur,
                """
                WITH document_tenants AS (
                    SELECT
                        pa.document_id,
                        array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    GROUP BY pa.document_id
                )
                SELECT
                    left_tenant AS tenant_a,
                    right_tenant AS tenant_b,
                    COUNT(*) AS shared_document_count
                FROM document_tenants
                CROSS JOIN LATERAL unnest(tenant_ids) AS left_tenant
                CROSS JOIN LATERAL unnest(tenant_ids) AS right_tenant
                WHERE left_tenant < right_tenant
                GROUP BY left_tenant, right_tenant
                ORDER BY shared_document_count DESC, tenant_a, tenant_b
                LIMIT 100;
                """,
            )
            if progress is not None:
                progress.update("computed tenant cooccurrence pairs")
    finally:
        conn.close()

    total_documents = max(int(scalar_counts["document_count"]), 1)
    total_vectors = max(int(scalar_counts["block_count"]), 1)
    tenant_rows.sort(
        key=lambda row: (
            -int(row.get("vector_count", 0) or 0),
            -int(row.get("document_count", 0) or 0),
            int(row.get("user_id", 0) or 0),
        )
    )
    for rank, row in enumerate(tenant_rows, start=1):
        document_count = int(row.get("document_count", 0) or 0)
        vector_count = int(row.get("vector_count", 0) or 0)
        row["vector_rank"] = int(rank)
        row["document_share"] = float(document_count / total_documents)
        row["vector_share"] = float(vector_count / total_vectors)
        row["vectors_per_document"] = float(vector_count / max(document_count, 1))

    acl_size_values = [int(row["acl_size"]) for row in acl_rows]
    acl_document_values = [int(row["document_count"]) for row in acl_rows]
    acl_vector_values = [int(row["vector_count"]) for row in acl_rows]
    tenant_document_values = [int(row["document_count"]) for row in tenant_rows]
    tenant_vector_values = [int(row["vector_count"]) for row in tenant_rows]
    tenant_role_values = [int(row["role_count"]) for row in tenant_rows]
    tenant_cooccurrence_values = [float(row["cooccurrence_count"]) for row in tenant_rows]
    role_tenant_values = [int(row["tenant_count"]) for row in role_rows]
    role_document_values = [int(row["document_count"]) for row in role_rows]

    summary = {
        **scalar_counts,
        "acl_count": int(len(acl_rows)),
        "tenant_document_distribution": _distribution(tenant_document_values),
        "tenant_vector_distribution": _distribution(tenant_vector_values),
        "tenant_role_distribution": _distribution(tenant_role_values),
        "tenant_cooccurrence_distribution": _distribution(tenant_cooccurrence_values),
        "acl_size_distribution": _distribution(acl_size_values),
        "acl_document_distribution": _distribution(acl_document_values),
        "acl_vector_distribution": _distribution(acl_vector_values),
        "role_tenant_distribution": _distribution(role_tenant_values),
        "role_document_distribution": _distribution(role_document_values),
    }

    return {
        "summary": summary,
        "tenant_stats": tenant_rows,
        "tenant_vector_counts": [
            {
                "user_id": row["user_id"],
                "vector_rank": row["vector_rank"],
                "vector_count": row["vector_count"],
                "vector_share": row["vector_share"],
                "document_count": row["document_count"],
                "document_share": row["document_share"],
                "vectors_per_document": row["vectors_per_document"],
                "acl_count": row["acl_count"],
                "role_count": row["role_count"],
                "cooccurrence_count": row["cooccurrence_count"],
                "avg_cooccurrence_per_document": row["avg_cooccurrence_per_document"],
            }
            for row in tenant_rows
        ],
        "acl_stats": acl_rows,
        "role_stats": role_rows,
        "top_tenant_pairs": pair_rows,
    }


def write_markdown_report(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    tenant_stats = result["tenant_stats"]
    acl_stats = result["acl_stats"]
    role_stats = result["role_stats"]
    top_pairs = result["top_tenant_pairs"]

    lines = [
        "# Permission Structure Debug Report",
        "",
        "## Summary",
        "",
        f"- Tenants: {summary['tenant_count']}",
        f"- Roles in UserRoles: {summary['role_count']}",
        f"- Roles in PermissionAssignment: {summary['permission_role_count']}",
        f"- ACL patterns: {summary['acl_count']}",
        f"- Documents in documentblocks: {summary['document_count']}",
        f"- Documents with permissions: {summary['permission_document_count']}",
        f"- Blocks / vectors: {summary['block_count']}",
        f"- User-role assignments: {summary['user_role_assignment_count']}",
        f"- Permission assignments: {summary['permission_assignment_count']}",
        "",
        "## Tenant Distributions",
        "",
        "| metric | mean | p50 | p90 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, title in [
        ("tenant_document_distribution", "documents per tenant"),
        ("tenant_vector_distribution", "vectors per tenant"),
        ("tenant_role_distribution", "roles per tenant"),
        ("tenant_cooccurrence_distribution", "cooccurrence per tenant"),
    ]:
        dist = summary[key]
        lines.append(
            f"| {title} | {dist['mean']:.2f} | {dist['p50']:.2f} | {dist['p90']:.2f} | {dist['p95']:.2f} | {dist['max']:.2f} |"
        )

    lines.extend([
        "",
        "## ACL Distributions",
        "",
        "| metric | mean | p50 | p90 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for key, title in [
        ("acl_size_distribution", "tenants per ACL"),
        ("acl_document_distribution", "documents per ACL"),
        ("acl_vector_distribution", "vectors per ACL"),
    ]:
        dist = summary[key]
        lines.append(
            f"| {title} | {dist['mean']:.2f} | {dist['p50']:.2f} | {dist['p90']:.2f} | {dist['p95']:.2f} | {dist['max']:.2f} |"
        )

    lines.extend([
        "",
        "## Top Tenants by Document Count",
        "",
        "| user_id | roles | documents | vectors | cooccurrence | avg coocc/doc |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in tenant_stats[:20]:
        lines.append(
            "| {user_id} | {role_count} | {document_count} | {vector_count} | {cooccurrence_count} | {avg_cooccurrence_per_document:.2f} |".format(
                **row
            )
        )

    lines.extend([
        "",
        "## Top ACL Patterns by Document Count",
        "",
        "| acl_id | acl_size | documents | vectors | tenant_ids |",
        "| ---: | ---: | ---: | ---: | --- |",
    ])
    for row in acl_stats[:20]:
        tenant_ids = ",".join(str(value) for value in row["tenant_ids"])
        lines.append(
            f"| {row['acl_id']} | {row['acl_size']} | {row['document_count']} | {row['vector_count']} | `{tenant_ids}` |"
        )

    lines.extend([
        "",
        "## Top Roles by Document Count",
        "",
        "| role_id | tenants | documents | vectors |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for row in role_stats[:20]:
        lines.append(
            f"| {row['role_id']} | {row['tenant_count']} | {row['document_count']} | {row['vector_count']} |"
        )

    lines.extend([
        "",
        "## Top Tenant Cooccurrence Pairs",
        "",
        "| tenant_a | tenant_b | shared_documents |",
        "| ---: | ---: | ---: |",
    ])
    for row in top_pairs[:30]:
        lines.append(
            f"| {row['tenant_a']} | {row['tenant_b']} | {row['shared_document_count']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=11)
    result = collect_permission_structure(progress=progress)

    (RESULT_DIR / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    progress.update("wrote summary.json")
    _write_csv(RESULT_DIR / "tenant_stats.csv", result["tenant_stats"])
    progress.update("wrote tenant_stats.csv")
    _write_csv(RESULT_DIR / "tenant_vector_counts.csv", result["tenant_vector_counts"])
    progress.update("wrote tenant_vector_counts.csv")
    _write_csv(RESULT_DIR / "acl_stats.csv", result["acl_stats"])
    progress.update("wrote acl_stats.csv")
    _write_csv(RESULT_DIR / "role_stats.csv", result["role_stats"])
    progress.update("wrote role_stats.csv")
    _write_csv(RESULT_DIR / "tenant_pair_cooccurrence_top100.csv", result["top_tenant_pairs"])
    write_markdown_report(RESULT_DIR / "summary.md", result)
    progress.update("wrote cooccurrence csv and summary.md")

    summary = result["summary"]
    print("Permission structure debug finished.")
    print(f"Result directory: {RESULT_DIR}")
    print(f"Tenants: {summary['tenant_count']}")
    print(f"Roles: {summary['role_count']}")
    print(f"ACL patterns: {summary['acl_count']}")
    print(f"Documents: {summary['document_count']}")
    print(f"Blocks/vectors: {summary['block_count']}")


if __name__ == "__main__":
    main()
