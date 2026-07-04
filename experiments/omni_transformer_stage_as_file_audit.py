from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visual_multimodal_stage_ab import write_png  # noqa: E402


ISSUES = ("html_csv_mismatch", "policy_threshold_violation", "screenshot_html_conflict", "clean")
CITATIONS = ("html+csv", "csv+pdf", "html+screenshot", "all-clear")
ACTIONS = ("READ_HTML", "READ_CSV", "READ_PDF", "READ_SCREENSHOT", "FINAL")
READ_ORDER = (0, 1, 2, 3)
HISTORY_LEN = 4
VALUE_COUNT = 4
HIST_PAD = 0
HIST_HTML_BASE = 1
HIST_CSV_BASE = HIST_HTML_BASE + VALUE_COUNT
HIST_PDF_BASE = HIST_CSV_BASE + VALUE_COUNT
HIST_SCREEN_BASE = HIST_PDF_BASE + VALUE_COUNT
HISTORY_VOCAB = HIST_SCREEN_BASE + VALUE_COUNT
FILE_TOKEN_COUNT = 4


@dataclass(frozen=True)
class StageASConfig:
    train_size: int = 2048
    val_size: int = 512
    test_size: int = 512
    batch_size: int = 128
    seed: int = 20260701
    d_model: int = 96
    heads: int = 4
    layers: int = 2
    latent_tokens: int = 6
    lr: float = 8e-4
    steps: int = 650
    eval_every: int = 200
    sample_count: int = 12


@dataclass(frozen=True)
class FileAuditExample:
    html_value: int
    csv_value: int
    pdf_threshold: int
    screenshot_value: int
    issue: int
    citation: int
    case_id: str


@dataclass(frozen=True)
class FileAuditSet:
    examples: list[FileAuditExample]
    file_tokens: torch.Tensor
    history: torch.Tensor
    issue: torch.Tensor
    citation: torch.Tensor

    def subset(self, indices: list[int]) -> "FileAuditSet":
        return FileAuditSet(
            examples=[self.examples[index] for index in indices],
            file_tokens=self.file_tokens[indices],
            history=self.history[indices],
            issue=self.issue[indices],
            citation=self.citation[indices],
        )

    def to(self, device: torch.device) -> "FileAuditSet":
        return FileAuditSet(
            examples=self.examples,
            file_tokens=self.file_tokens.to(device=device),
            history=self.history.to(device=device),
            issue=self.issue.to(device=device),
            citation=self.citation.to(device=device),
        )


class StatusHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.value: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "div" and attrs_dict.get("id") == "reported-status":
            raw = attrs_dict.get("data-status")
            if raw is not None:
                self.value = int(raw)


def issue_for(html_value: int, csv_value: int, pdf_threshold: int, screenshot_value: int) -> int:
    if html_value != csv_value:
        return 0
    if csv_value > pdf_threshold:
        return 1
    if screenshot_value != html_value:
        return 2
    return 3


def history_tokens(example: FileAuditExample) -> torch.Tensor:
    return torch.tensor(
        [
            HIST_HTML_BASE + example.html_value,
            HIST_CSV_BASE + example.csv_value,
            HIST_PDF_BASE + example.pdf_threshold,
            HIST_SCREEN_BASE + example.screenshot_value,
        ],
        dtype=torch.long,
    )


def make_example(index: int, *, rng: random.Random) -> FileAuditExample:
    issue_target = index % len(ISSUES)
    if issue_target == 0:
        html_value = rng.randrange(VALUE_COUNT)
        csv_value = (html_value + 1 + rng.randrange(VALUE_COUNT - 1)) % VALUE_COUNT
        pdf_threshold = max(html_value, csv_value)
        screenshot_value = html_value
    elif issue_target == 1:
        html_value = rng.randrange(1, VALUE_COUNT)
        csv_value = html_value
        pdf_threshold = rng.randrange(html_value)
        screenshot_value = html_value
    elif issue_target == 2:
        html_value = rng.randrange(VALUE_COUNT)
        csv_value = html_value
        pdf_threshold = max(html_value, rng.randrange(VALUE_COUNT))
        screenshot_value = (html_value + 1 + rng.randrange(VALUE_COUNT - 1)) % VALUE_COUNT
    else:
        html_value = rng.randrange(VALUE_COUNT)
        csv_value = html_value
        pdf_threshold = max(html_value, rng.randrange(VALUE_COUNT))
        screenshot_value = html_value
    issue = issue_for(html_value, csv_value, pdf_threshold, screenshot_value)
    return FileAuditExample(
        html_value=html_value,
        csv_value=csv_value,
        pdf_threshold=pdf_threshold,
        screenshot_value=screenshot_value,
        issue=issue,
        citation=issue,
        case_id=f"case-{index:05d}",
    )


def build_split(split: str, size: int, config: StageASConfig) -> FileAuditSet:
    split_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    rng = random.Random(config.seed + split_offsets[split])
    examples = [make_example(index, rng=rng) for index in range(size)]
    return FileAuditSet(
        examples=examples,
        file_tokens=torch.arange(FILE_TOKEN_COUNT, dtype=torch.long).view(1, FILE_TOKEN_COUNT).expand(size, -1).clone(),
        history=torch.stack([history_tokens(example) for example in examples]),
        issue=torch.tensor([example.issue for example in examples], dtype=torch.long),
        citation=torch.tensor([example.citation for example in examples], dtype=torch.long),
    )


def random_batch(data: FileAuditSet, *, rng: random.Random, batch_size: int, device: torch.device) -> FileAuditSet:
    return data.subset([rng.randrange(len(data.examples)) for _ in range(batch_size)]).to(device)


def render_screenshot(value: int) -> torch.Tensor:
    image = torch.full((3, 64, 64), 0.07)
    image[:, 6:58, 6:58] = 0.12
    colors = torch.tensor(
        [
            [0.88, 0.10, 0.10],
            [0.10, 0.70, 0.25],
            [0.15, 0.32, 0.90],
            [0.94, 0.74, 0.12],
        ]
    )
    image[:, 20:44, 20:44] = colors[value].view(3, 1, 1)
    image[:, 50:53, 8 + value * 12 : 17 + value * 12] = 0.95
    return image


def write_minimal_pdf(path: Path, text: str) -> None:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"5 0 obj << /Length " + str(len(stream)).encode("ascii") + b" >> stream\n" + stream + b"\nendstream endobj\n",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(body))
        body += obj
    xref_offset = len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n" + b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
    trailer = f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    path.write_bytes(body + xref + trailer)


def write_case_files(example: FileAuditExample, root: Path) -> dict[str, str]:
    case_dir = root / example.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    html = case_dir / "dashboard.html"
    csv_path = case_dir / "metrics.csv"
    pdf = case_dir / "policy.pdf"
    screenshot = case_dir / "screenshot.png"
    meta = case_dir / "screenshot.json"

    html.write_text(
        f"<html><body><div id=\"reported-status\" data-status=\"{example.html_value}\">Status {example.html_value}</div></body></html>\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "actual_status", "value": example.csv_value})
    write_minimal_pdf(pdf, f"policy threshold={example.pdf_threshold}")
    write_png(screenshot, render_screenshot(example.screenshot_value))
    meta.write_text(json.dumps({"status_badge": example.screenshot_value}, indent=2), encoding="utf-8")
    return {"html": str(html), "csv": str(csv_path), "pdf": str(pdf), "screenshot": str(screenshot), "screenshot_meta": str(meta)}


def read_html_status(path: Path) -> int:
    parser = StatusHTMLParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if parser.value is None:
        raise ValueError(f"missing html status in {path}")
    return parser.value


def read_csv_status(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("metric") == "actual_status":
            return int(row["value"])
    raise ValueError(f"missing csv actual_status in {path}")


def read_pdf_threshold(path: Path) -> int:
    match = re.search(rb"threshold=(\d)", path.read_bytes())
    if not match:
        raise ValueError(f"missing pdf threshold in {path}")
    return int(match.group(1))


def read_screenshot_status(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return int(data["status_badge"])


def build_tool_history(files: dict[str, str]) -> list[int]:
    return [
        HIST_HTML_BASE + read_html_status(Path(files["html"])),
        HIST_CSV_BASE + read_csv_status(Path(files["csv"])),
        HIST_PDF_BASE + read_pdf_threshold(Path(files["pdf"])),
        HIST_SCREEN_BASE + read_screenshot_status(Path(files["screenshot_meta"])),
    ]


def write_sample_files(data: FileAuditSet, path: Path, limit: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for example in data.examples[:limit]:
        files = write_case_files(example, path)
        tool_history = build_tool_history(files)
        rows.append(
            {
                "case_id": example.case_id,
                "files": files,
                "tool_history": tool_history,
                "issue": ISSUES[example.issue],
                "citation": CITATIONS[example.citation],
                "audit_report": f"{ISSUES[example.issue]} cited by {CITATIONS[example.citation]}",
            }
        )
    (path / "sample_audits.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


class FileAuditLatentModel(nn.Module):
    def __init__(self, config: StageASConfig) -> None:
        super().__init__()
        self.file_embed = nn.Embedding(FILE_TOKEN_COUNT, config.d_model)
        self.history_embed = nn.Embedding(HISTORY_VOCAB, config.d_model, padding_idx=HIST_PAD)
        self.type_embed = nn.Embedding(3, config.d_model)
        self.latent = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.issue_head = nn.Linear(config.d_model, len(ISSUES))
        self.citation_head = nn.Linear(config.d_model, len(CITATIONS))
        self.action_head = nn.Linear(config.d_model, len(ACTIONS))

    def forward(self, batch: FileAuditSet, *, no_tool_history: bool = False) -> dict[str, torch.Tensor]:
        history = torch.full_like(batch.history, HIST_PAD) if no_tool_history else batch.history
        file_tokens = self.file_embed(batch.file_tokens) + self.type_embed.weight[0].view(1, 1, -1)
        history_tokens = self.history_embed(history) + self.type_embed.weight[1].view(1, 1, -1)
        latent = self.latent.unsqueeze(0).expand(batch.history.shape[0], -1, -1) + self.type_embed.weight[2].view(1, 1, -1)
        encoded = self.encoder(torch.cat((file_tokens, history_tokens, latent), dim=1))
        pooled = self.norm(encoded[:, -latent.shape[1] :]).mean(dim=1)
        return {
            "issue": self.issue_head(pooled),
            "citation": self.citation_head(pooled),
            "action": self.action_head(pooled),
        }


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().item())


def train_model(model: FileAuditLatentModel, train: FileAuditSet, val: FileAuditSet, config: StageASConfig, device: torch.device) -> dict[str, object]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    rng = random.Random(config.seed + 101)
    history = []
    for step in range(1, config.steps + 1):
        model.train()
        batch = random_batch(train, rng=rng, batch_size=config.batch_size, device=device)
        outputs = model(batch)
        loss = (
            F.cross_entropy(outputs["issue"], batch.issue)
            + F.cross_entropy(outputs["citation"], batch.citation)
            + 0.2 * F.cross_entropy(outputs["action"], torch.full_like(batch.issue, 4))
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate(model, val, device, prefix="val")
            metrics["step"] = float(step)
            metrics["loss"] = float(loss.detach().item())
            history.append(metrics)
            print(json.dumps(metrics), flush=True)
    return {"history": history}


@torch.no_grad()
def evaluate(model: FileAuditLatentModel, data: FileAuditSet, device: torch.device, *, prefix: str) -> dict[str, float]:
    model.eval()
    batch = data.to(device)
    full = model(batch)
    no_history = model(batch, no_tool_history=True)
    issue_pred = full["issue"].argmax(dim=-1)
    citation_pred = full["citation"].argmax(dim=-1)
    action_pred = full["action"].argmax(dim=-1)
    both = ((issue_pred == batch.issue) & (citation_pred == batch.citation)).float().mean().item()
    return {
        f"{prefix}_conclusion_accuracy": accuracy(full["issue"], batch.issue),
        f"{prefix}_citation_accuracy": accuracy(full["citation"], batch.citation),
        f"{prefix}_report_exact": float(both),
        f"{prefix}_tool_step_legality": float((action_pred == 4).float().mean().item()),
        f"{prefix}_no_tool_history_conclusion_accuracy": accuracy(no_history["issue"], batch.issue),
        f"{prefix}_no_tool_history_citation_accuracy": accuracy(no_history["citation"], batch.citation),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_one(config: StageASConfig, output_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = build_split("train", config.train_size, config)
    val = build_split("val", config.val_size, config)
    test = build_split("test", config.test_size, config)
    write_sample_files(test, output_path.parent / "sample_files", config.sample_count)
    model = FileAuditLatentModel(config)
    training = train_model(model, train, val, config, device)
    metrics = evaluate(model, test, device, prefix="test")
    result = {
        "schema_version": 1,
        "stage": "AS",
        "config": asdict(config),
        "device": str(device),
        "metrics": metrics,
        "cost": {"elapsed_sec": time.perf_counter() - started, "parameters": parameter_count(model)},
        "training": training,
        "interpretation": {
            "goal": "Audit real local file boundaries through controlled HTML/CSV/PDF/screenshot tools and emit cited conclusions.",
            "success_condition": "Conclusion, citation, and tool legality >= 85%, with no-tool-history clearly worse.",
            "boundary": "Synthetic file contents and metadata-backed screenshot status; this does not prove OCR/layout robustness.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return result


def collect_numbers(value: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            out.update(collect_numbers(child, name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    rows = [collect_numbers(run) for run in runs]
    keys = sorted(set().union(*(row.keys() for row in rows)))
    summary = {key: statistics.mean(row[key] for row in rows if key in row) for key in keys}
    stdev = {
        key: statistics.pstdev([row[key] for row in rows if key in row])
        for key in keys
        if sum(1 for row in rows if key in row) > 1
    }
    no_history_gap = summary["metrics_test_conclusion_accuracy"] - summary["metrics_test_no_tool_history_conclusion_accuracy"]
    return {
        "schema_version": 1,
        "stage": "AS",
        "runs": runs,
        "summary": summary,
        "stdev": stdev,
        "gates": {
            "conclusion_accuracy": summary["metrics_test_conclusion_accuracy"],
            "citation_accuracy": summary["metrics_test_citation_accuracy"],
            "report_exact": summary["metrics_test_report_exact"],
            "tool_step_legality": summary["metrics_test_tool_step_legality"],
            "no_tool_history_gap": no_history_gap,
            "passes_primary_85": all(
                summary[key] >= 0.85
                for key in (
                    "metrics_test_conclusion_accuracy",
                    "metrics_test_citation_accuracy",
                    "metrics_test_tool_step_legality",
                )
            ),
            "passes_no_tool_history_gap_30pp": no_history_gap >= 0.30,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seeds", default="20260701,20260702,20260703")
    parser.add_argument("--output", type=Path, default=Path("artifacts/omni_transformer_stage_as_file_audit/result.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/omni_transformer_stage_as_file_audit/sweep_runs"))
    parser.add_argument("--aggregate", type=Path, default=Path("artifacts/omni_transformer_stage_as_file_audit/sweep_results.json"))
    parser.add_argument("--train-size", type=int, default=StageASConfig.train_size)
    parser.add_argument("--val-size", type=int, default=StageASConfig.val_size)
    parser.add_argument("--test-size", type=int, default=StageASConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=StageASConfig.batch_size)
    parser.add_argument("--d-model", type=int, default=StageASConfig.d_model)
    parser.add_argument("--heads", type=int, default=StageASConfig.heads)
    parser.add_argument("--layers", type=int, default=StageASConfig.layers)
    parser.add_argument("--latent-tokens", type=int, default=StageASConfig.latent_tokens)
    parser.add_argument("--steps", type=int, default=StageASConfig.steps)
    parser.add_argument("--eval-every", type=int, default=StageASConfig.eval_every)
    parser.add_argument("--sample-count", type=int, default=StageASConfig.sample_count)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, seed: int) -> StageASConfig:
    return StageASConfig(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        seed=seed,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        latent_tokens=args.latent_tokens,
        steps=args.steps,
        eval_every=args.eval_every,
        sample_count=args.sample_count,
    )


def main() -> None:
    args = parse_args()
    if args.sweep:
        runs = []
        for seed_text in args.seeds.split(","):
            seed = int(seed_text.strip())
            runs.append(run_one(config_from_args(args, seed), args.output_dir / f"seed{seed}.json"))
        aggregate = aggregate_runs(runs)
        args.aggregate.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"aggregate": aggregate["gates"]}, ensure_ascii=False, indent=2), flush=True)
    else:
        first_seed = int(args.seeds.split(",")[0].strip())
        run_one(config_from_args(args, first_seed), args.output)


if __name__ == "__main__":
    main()
