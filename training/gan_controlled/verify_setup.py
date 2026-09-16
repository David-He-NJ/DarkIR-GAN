"""Static, dependency-free checks for the controlled GAN setup."""

import ast
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASELINE = HERE.parent / "baseline_continued"


def fail(message):
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_python(path):
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        fail(f"syntax error in {path.relative_to(REPO)}: {exc}")


def main():
    controlled_script = HERE / "train_controlled_gan.py"
    patchgan_script = HERE / "patchgan.py"
    config_path = HERE / "options" / "config.yaml"

    parse_python(controlled_script)
    parse_python(patchgan_script)
    parse_python(Path(__file__))

    script = controlled_script.read_text(encoding="utf-8")
    baseline_script = (BASELINE / "train_continued.py").read_text(encoding="utf-8")
    required_fragments = (
        "baseline.create_model",
        "baseline.create_dataloaders",
        "baseline.create_loss",
        "baseline.create_optim_scheduler",
        "baseline.validate_quick",
        "baseline.validate",
        "criterion[\"enhance_loss\"](outputs, targets)",
        "reconstruction + adversarial_opt[\"weight\"] * loss_adv",
        "generator.load_state_dict(checkpoint[\"model_state_dict\"], strict=True)",
        "optimizer_g.load_state_dict(checkpoint[\"optimizer_state_dict\"])",
        "scheduler_g.load_state_dict(checkpoint[\"scheduler_state_dict\"])",
    )
    for fragment in required_fragments:
        if fragment not in script:
            fail(f"missing controlled-invariant fragment: {fragment}")

    forbidden_fragments = (
        "criterion['enhance_loss'](inputs, targets)",
        'criterion["enhance_loss"](inputs, targets)',
        "torch.optim.Adam(generator.parameters()",
        "betas=(0.5, 0.999))\n    optimizer_g",
    )
    for fragment in forbidden_fragments:
        if fragment in script:
            fail(f"forbidden generator-side fragment found: {fragment}")

    required_baseline_fragments = (
        "TOTAL_EPOCHS = 30",
        "LEARNING_RATE = 1e-4",
        "ETA_MIN = 1e-6",
        "criterion['enhance_loss'](outputs, targets)",
        "0.1 * loss_perceptual",
        "5.0 * loss_edge",
        "0.5 * loss_enhance",
        "optimizer.load_state_dict(checkpoint['optimizer_state_dict'])",
        "scheduler.load_state_dict(checkpoint['scheduler_state_dict'])",
    )
    for fragment in required_baseline_fragments:
        if fragment not in baseline_script:
            fail(f"baseline_continued runtime assumption changed: {fragment}")

    config = config_path.read_text(encoding="utf-8")
    marker = "# Controlled GAN additions."
    if marker not in config:
        fail("controlled config marker is missing")
    baseline_prefix, additions = config.split(marker, 1)
    baseline_config = (BASELINE / "options" / "config.yaml").read_text(encoding="utf-8")

    # Comments differ slightly to explain control semantics; compare normalized
    # non-comment YAML lines so the baseline-controlled values remain identical.
    def normalized_yaml_lines(text):
        result = []
        for line in text.splitlines():
            content = line.split("#", 1)[0].rstrip()
            if content:
                result.append(content)
        return result

    if normalized_yaml_lines(baseline_prefix) != normalized_yaml_lines(baseline_config):
        fail("controlled generator/data/train YAML differs from baseline_continued")
    if "adversarial:" not in additions or "discriminator:" not in additions:
        fail("controlled config additions are incomplete")

    print("CONTROLLED_GAN_STATIC_CHECK_OK")


if __name__ == "__main__":
    main()
