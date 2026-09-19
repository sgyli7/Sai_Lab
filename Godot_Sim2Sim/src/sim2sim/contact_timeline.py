"""Read-only MJ vs GD contact timeline for kick (first ~1 s). Does not change plant.

Per control tick (20 ms): foot contact counts / impulses (what each backend exposes),
ankle COM z, trunk tilt, key joint q/qd. Aligns first gate fall (~0.6 s) and finds
earliest contact divergence (target t<0.3 s).

Godot: headless step replies already include per-body dump (n_contacts, impulse, com).
MuJoCo: data.ncon + mj_contactForce; ankle COM from body xpos.

Usage:
  cd sim2sim && uv run python -m sim2sim.contact_timeline
  uv run python -m sim2sim.contact_timeline --skill kick_right --seconds 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from sim2sim.coords import quat_rotate_inverse_wxyz
from sim2sim.godot_proc import sim2sim_root
from sim2sim.obs import DEFAULT_HOME, build_obs, command_13
from sim2sim.paths import policies_dir
from sim2sim.policy import OnnxPolicy
from sim2sim.runner import apply_home_qpos, load_robot_cfg, make_backend
from sim2sim.skill_metrics import FALL_TILT, FALL_Z, tilt_deg_from_quat

ROOT = sim2sim_root()
POL = policies_dir()

ONNX = {
    "kick_left": POL / "ball_kick_left.onnx",
    "kick_right": POL / "ball_kick_right.onnx",
}

# actuator indices
L_HIP_P, L_KNEE, L_ANKLE = 2, 3, 4
R_HIP_P, R_KNEE, R_ANKLE = 11, 12, 13

FLOOR_NAMES = {"floor", "Floor", "StaticBody", "static_floor", "ground"}


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(name)
    return int(bid)


def _geom_body(model: mujoco.MjModel, gid: int) -> str:
    if gid < 0:
        return "world"
    bid = int(model.geom_bodyid[gid])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body{bid}"


def mj_foot_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Per-foot contact count, normal force sum, impulse proxy, self-contact flag."""
    left_n = right_n = 0
    left_fn = right_fn = 0.0
    left_imp = right_imp = 0.0
    self_n = 0
    other_n = 0
    force = np.zeros(6, dtype=np.float64)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        b1, b2 = _geom_body(model, g1), _geom_body(model, g2)
        names = {b1, b2}
        mujoco.mj_contactForce(model, data, i, force)
        # force[0] is normal (constraint frame); |force[:3]| ≈ contact wrench linear
        fn = abs(float(force[0]))
        fmag = float(np.linalg.norm(force[:3]))
        is_floor = ("world" in names) or any(
            "floor" in n.lower() or n == "world" for n in names
        )
        # MuJoCo plane geom is usually on world body id 0 → name may be None→body0 or ""
        if b1 in ("", "body0") or b2 in ("", "body0"):
            is_floor = True
        hit_left = "ankle_left" in names
        hit_right = "ankle_right" in names
        if hit_left and hit_right:
            self_n += 1
            continue
        if hit_left and is_floor:
            left_n += 1
            left_fn += fn
            left_imp += fmag
        elif hit_right and is_floor:
            right_n += 1
            right_fn += fn
            right_imp += fmag
        elif hit_left or hit_right:
            # foot vs non-floor (swing self-hit / other body)
            other_n += 1
            if hit_left:
                left_n += 1
                left_fn += fn
                left_imp += fmag
            if hit_right:
                right_n += 1
                right_fn += fn
                right_imp += fmag
        else:
            other_n += 1
    return {
        "ncon": int(data.ncon),
        "L_n": left_n,
        "R_n": right_n,
        "L_fn": left_fn,
        "R_fn": right_fn,
        "L_imp": left_imp,
        "R_imp": right_imp,
        "self_n": self_n,
        "other_n": other_n,
    }


def gd_foot_contacts(dump: list[dict]) -> dict:
    """Parse Godot headless dump for ankle_left / ankle_right."""
    by = {d.get("name"): d for d in dump if isinstance(d, dict)}
    out = {
        "ncon": 0,
        "L_n": 0,
        "R_n": 0,
        "L_fn": 0.0,
        "R_fn": 0.0,
        "L_imp": 0.0,
        "R_imp": 0.0,
        "self_n": 0,
        "other_n": 0,
        "L_com_z": float("nan"),
        "R_com_z": float("nan"),
    }
    for side, key in (("L", "ankle_left"), ("R", "ankle_right")):
        d = by.get(key)
        if not d:
            continue
        com = d.get("com") or d.get("pos") or [0, 0, 0]
        out[f"{side}_com_z"] = float(com[2])
        n = int(d.get("n_contacts", 0))
        imp = float(d.get("impulse", 0.0))
        out[f"{side}_n"] = n
        out[f"{side}_imp"] = imp
        # Godot reports impulse length sum, not separated normal; use as force proxy
        out[f"{side}_fn"] = imp
        out["ncon"] += n
        whos = d.get("cwho") or []
        for w in whos:
            ws = str(w)
            if "ankle_" in ws and ws != key:
                out["self_n"] += 1
            elif ws not in FLOOR_NAMES and "floor" not in ws.lower() and ws not in ("?", ""):
                # non-floor collider (sibling heel, jaw, etc.)
                if "heel" in ws or "foot" in ws or "ankle" in ws:
                    pass
                else:
                    out["other_n"] += 1
    return out


def mj_ankle_com_z(backend) -> tuple[float, float]:
    m, d = backend.model, backend.data
    zl = float(d.xpos[_body_id(m, "ankle_left")][2])
    zr = float(d.xpos[_body_id(m, "ankle_right")][2])
    return zl, zr


def row_from_state(
    *,
    t: float,
    q: np.ndarray,
    qd: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    contacts: dict,
    L_com_z: float,
    R_com_z: float,
) -> dict:
    tilt = float(tilt_deg_from_quat(base_quat.reshape(1, 4))[0])
    g = quat_rotate_inverse_wxyz(base_quat, np.array([0.0, 0.0, -1.0]))
    gate_fell = bool(base_pos[2] < FALL_Z or tilt > FALL_TILT)
    return {
        "t": float(t),
        "z": float(base_pos[2]),
        "tilt_deg": tilt,
        "grav_z": float(g[2]),
        "gate_fell": gate_fell,
        "xy": float(np.linalg.norm(base_pos[:2])),
        "L_n": contacts["L_n"],
        "R_n": contacts["R_n"],
        "L_fn": contacts["L_fn"],
        "R_fn": contacts["R_fn"],
        "L_imp": contacts["L_imp"],
        "R_imp": contacts["R_imp"],
        "ncon": contacts["ncon"],
        "self_n": contacts.get("self_n", 0),
        "other_n": contacts.get("other_n", 0),
        "L_com_z": float(L_com_z),
        "R_com_z": float(R_com_z),
        "q_Lhip": float(q[L_HIP_P]),
        "q_Lknee": float(q[L_KNEE]),
        "q_Lankle": float(q[L_ANKLE]),
        "q_Rhip": float(q[R_HIP_P]),
        "q_Rknee": float(q[R_KNEE]),
        "q_Rankle": float(q[R_ANKLE]),
        "qd_Lhip": float(qd[L_HIP_P]),
        "qd_Lknee": float(qd[L_KNEE]),
        "qd_Rhip": float(qd[R_HIP_P]),
        "qd_Rknee": float(qd[R_KNEE]),
        "qd_Lankle": float(qd[L_ANKLE]),
        "qd_Rankle": float(qd[R_ANKLE]),
    }


def rollout_contact(backend, policy, cfg: dict, *, seconds: float) -> list[dict]:
    home = np.asarray(cfg.get("home", DEFAULT_HOME), dtype=np.float32)
    scale = float(cfg.get("action_scale", 1.0))
    decimation = int(cfg.get("decimation", 4))
    z0 = float(cfg.get("reset_z", 0.125))
    command = command_13(np.zeros(3, dtype=np.float32))

    if backend.name == "mujoco":
        apply_home_qpos(backend, home, z=z0)
        st = backend.reset(qpos=backend.data.qpos.copy(), qvel=backend.data.qvel.copy(), ctrl=home)
    else:
        from sim2sim.backends.mujoco_backend import MujocoBackend

        mj = MujocoBackend(Path(cfg["mjcf"]), timestep=cfg.get("timestep", 0.005), current_limit_a=0.0)
        apply_home_qpos(mj, home, z=z0)
        poses = mj.body_poses_mujoco()
        mj.close()
        st = backend.reset(ctrl=home, bodies=poses)

    last_action = np.zeros(policy.act_dim, dtype=np.float32)
    n = int(round(seconds / (decimation * backend.dt)))
    rows: list[dict] = []

    for _ in range(n):
        obs = build_obs(st, last_action, command, home=home)
        action = policy.infer(obs)
        last_action = action.copy()
        ctrl = home + action * scale
        st = backend.step(ctrl, n_substeps=decimation)

        if backend.name == "mujoco":
            contacts = mj_foot_contacts(backend.model, backend.data)
            Lz, Rz = mj_ankle_com_z(backend)
        else:
            raw = st.extra.get("raw") or {}
            dump = raw.get("dump") or []
            contacts = gd_foot_contacts(dump)
            Lz = contacts.pop("L_com_z", float("nan"))
            Rz = contacts.pop("R_com_z", float("nan"))
            # restore keys used by row_from_state
            contacts.setdefault("L_n", 0)

        rows.append(
            row_from_state(
                t=st.t,
                q=st.q,
                qd=st.qd,
                base_pos=st.base_pos,
                base_quat=st.base_quat_wxyz,
                contacts=contacts,
                L_com_z=Lz,
                R_com_z=Rz,
            )
        )
    return rows


def first_gate_fall(rows: list[dict]) -> float | None:
    for r in rows:
        if r["gate_fell"]:
            return float(r["t"])
    return None


def find_divergence(mj: list[dict], gd: list[dict], *, support: str) -> dict:
    """Earliest *meaningful* contact divergence (ignore t<0.10 settling / reduction flicker).

    support = 'R' for kick_left, 'L' for kick_right.
    Meaningful = GD support n=0 for >=2 consecutive control ticks while MJ still contacting.
    """
    n = min(len(mj), len(gd))
    swing = "L" if support == "R" else "R"
    events: list[dict] = []
    structural: list[dict] = []
    first_sustained = None
    sustain = 0
    swing_recontact = None
    tilt_div = None

    for i in range(n):
        a, b = mj[i], gd[i]
        t = float(a["t"])
        sa, sb = int(a[f"{support}_n"]), int(b[f"{support}_n"])
        if t < 0.10:
            if sb >= 3 and sa <= 1:
                structural.append(
                    {"t": t, "note": "GD_full_pad_vs_MJ_reduction", "MJ_n": sa, "GD_n": sb}
                )
            continue
        notes: list[str] = []
        if sa > 0 and sb == 0:
            sustain += 1
            notes.append(f"support_{support}_lost_GD_tick{sustain}")
            if sustain >= 2 and first_sustained is None:
                first_sustained = {
                    "t": t,
                    "notes": [
                        f"support_{support}_lost_GD_sustained",
                        f"com_z MJ={a[f'{support}_com_z']:.4f} GD={b[f'{support}_com_z']:.4f}",
                        f"tilt MJ={a['tilt_deg']:.1f} GD={b['tilt_deg']:.1f}",
                    ],
                    "MJ": {k: a[k] for k in ("L_n", "R_n", "L_com_z", "R_com_z", "tilt_deg", "z")},
                    "GD": {k: b[k] for k in ("L_n", "R_n", "L_com_z", "R_com_z", "tilt_deg", "z")},
                }
        else:
            sustain = 0
        if (
            swing_recontact is None
            and 0.15 <= t < 0.5
            and int(b[f"{swing}_n"]) > 0
            and int(a[f"{swing}_n"]) == 0
        ):
            swing_recontact = {"t": t, "GD_n": int(b[f"{swing}_n"]), "MJ_n": 0}
            notes.append(f"swing_{swing}_recontact_GD")
        dtilt = float(b["tilt_deg"]) - float(a["tilt_deg"])
        if tilt_div is None and abs(dtilt) > 8.0:
            tilt_div = {"t": t, "dtilt": dtilt}
            notes.append(f"tilt_delta={dtilt:.1f}")
        if notes:
            events.append(
                {
                    "t": t,
                    "notes": notes,
                    "MJ": {k: a[k] for k in ("L_n", "R_n", "L_com_z", "R_com_z", "tilt_deg", "z")},
                    "GD": {k: b[k] for k in ("L_n", "R_n", "L_com_z", "R_com_z", "tilt_deg", "z")},
                }
            )

    primary = "unknown"
    first = first_sustained
    if first is not None:
        primary = "support_foot_lost_contact"
    elif swing_recontact is not None:
        primary = "swing_leg_early_recontact"
        first = {"t": swing_recontact["t"], "notes": ["swing_recontact"], **swing_recontact}
    elif tilt_div is not None:
        primary = "tilt_divergence_possible_sideslip"
        first = {"t": tilt_div["t"], "notes": [f"tilt_delta={tilt_div['dtilt']:.1f}"], **tilt_div}

    return {
        "first_event_t": None if first is None else first["t"],
        "first_event": first,
        "primary_class": primary,
        "structural_early": structural[:5],
        "swing_recontact": swing_recontact,
        "tilt_div_gt8": tilt_div,
        "n_events": len(events),
        "events_first_10": events[:10],
        "note": "Ignores t<0.10 MJ reduction flicker; sustained = GD support n=0 for >=2 ticks",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--robot", type=Path, default=ROOT / "robots/microduck.json")
    p.add_argument("--skill", choices=list(ONNX.keys()), default="kick_left")
    p.add_argument("--also-right", action="store_true", help="also dump kick_right")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=ROOT / "results/skill_repro/contact_timeline")
    args = p.parse_args(argv)

    cfg = load_robot_cfg(args.robot)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    skills = [args.skill]
    if args.also_right and "kick_right" not in skills:
        skills.append("kick_right")

    summary: dict = {"skills": {}}

    for skill in skills:
        # kick_left → right foot support; kick_right → left support
        support = "R" if skill == "kick_left" else "L"
        policy = OnnxPolicy(ONNX[skill])
        backend_rows: dict[str, list[dict]] = {}
        for backend_name in ("mujoco", "godot"):
            print(f"=== contact timeline {skill} / {backend_name} ===", flush=True)
            backend = make_backend(backend_name, cfg, headless=True)
            try:
                rows = rollout_contact(backend, policy, cfg, seconds=args.seconds)
            finally:
                backend.close()
            backend_rows[backend_name] = rows
            write_csv(out / f"{skill}_{backend_name}.csv", rows)
            (out / f"{skill}_{backend_name}.json").write_text(json.dumps(rows, indent=2))

        mj, gd = backend_rows["mujoco"], backend_rows["godot"]
        div = find_divergence(mj, gd, support=support)
        fall_mj = first_gate_fall(mj)
        fall_gd = first_gate_fall(gd)
        skill_sum = {
            "skill": skill,
            "support_foot": support,
            "seconds": args.seconds,
            "fall_t_gate_mujoco": fall_mj,
            "fall_t_gate_godot": fall_gd,
            "divergence": div,
            "api_notes": {
                "mujoco": "ncon + mj_contactForce (normal force[0], |f[:3]| as impulse proxy); ankle COM = xpos",
                "godot": "headless dump n_contacts + impulse length sum; ankle COM from dump.com; no separate normal",
            },
        }
        summary["skills"][skill] = skill_sum
        (out / f"{skill}_divergence.json").write_text(json.dumps(skill_sum, indent=2))

        # merged side-by-side CSV for first 1s
        merged = []
        for i in range(min(len(mj), len(gd))):
            a, b = mj[i], gd[i]
            merged.append(
                {
                    "t": a["t"],
                    "MJ_L_n": a["L_n"],
                    "GD_L_n": b["L_n"],
                    "MJ_R_n": a["R_n"],
                    "GD_R_n": b["R_n"],
                    "MJ_L_imp": a["L_imp"],
                    "GD_L_imp": b["L_imp"],
                    "MJ_R_imp": a["R_imp"],
                    "GD_R_imp": b["R_imp"],
                    "MJ_L_com_z": a["L_com_z"],
                    "GD_L_com_z": b["L_com_z"],
                    "MJ_R_com_z": a["R_com_z"],
                    "GD_R_com_z": b["R_com_z"],
                    "MJ_tilt": a["tilt_deg"],
                    "GD_tilt": b["tilt_deg"],
                    "MJ_z": a["z"],
                    "GD_z": b["z"],
                    "MJ_q_Lhip": a["q_Lhip"],
                    "GD_q_Lhip": b["q_Lhip"],
                    "MJ_q_Rhip": a["q_Rhip"],
                    "GD_q_Rhip": b["q_Rhip"],
                    "MJ_qd_Lhip": a["qd_Lhip"],
                    "GD_qd_Lhip": b["qd_Lhip"],
                    "GD_gate_fell": int(b["gate_fell"]),
                }
            )
        write_csv(out / f"{skill}_merged.csv", merged)

        print(
            json.dumps(
                {
                    "skill": skill,
                    "support": support,
                    "fall_gd": fall_gd,
                    "first_div_t": div["first_event_t"],
                    "primary": div["primary_class"],
                    "first": div["first_event"],
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )

    # markdown conclusion
    lines = ["# Contact timeline (kick)", ""]
    for skill, s in summary["skills"].items():
        d = s["divergence"]
        lines += [
            f"## {skill}",
            f"- support foot: **{s['support_foot']}**",
            f"- gate fall GD: `{s['fall_t_gate_godot']}`  MJ: `{s['fall_t_gate_mujoco']}`",
            f"- earliest divergence t: **{d['first_event_t']}**",
            f"- primary class: **{d['primary_class']}**",
            "",
        ]
        if d["first_event"]:
            lines.append("```json")
            lines.append(json.dumps(d["first_event"], indent=2))
            lines.append("```")
            lines.append("")
    lines += [
        "## API limits",
        "- MuJoCo: true contact normal via `mj_contactForce`; count from `ncon`.",
        "- Godot/Jolt: `get_contact_impulse().length()` sum per body (not separated normal); "
        "floor vs non-floor via collider name when available.",
        "- If impulse scale differs across engines, prefer contact **counts** + ankle COM z + tilt as primary diverge signals.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"CONTACT_TIMELINE: wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
