"""Accepted Sai 001 free-driving configuration shared by scene entry points."""
import argparse

DEFAULT_DRIVE_SPEED = .5


def drive_speed(value: str) -> float:
    speed = float(value)
    if not .02 <= speed <= DEFAULT_DRIVE_SPEED:
        raise argparse.ArgumentTypeError("Flat cruise speed must be between 0.02 and 0.5 m/s")
    return speed
