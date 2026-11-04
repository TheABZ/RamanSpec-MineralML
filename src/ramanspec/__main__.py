"""Command line interface: python -m ramanspec demo --output results."""
import argparse
import json
from .pipeline import run_demo
from .simulation import SimulationConfig


def main():
    parser = argparse.ArgumentParser(description='Synthetic Raman research demonstration')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Run the complete simulation and analysis')
    demo.add_argument('--output', default='results')
    demo.add_argument('--seed', type=int, default=42)
    demo.add_argument('--noise', type=float, default=.025)
    demo.add_argument('--components', type=int, default=5)
    demo.add_argument('--clusters', type=int, default=3)
    demo.add_argument('--skip-noise-sweep', action='store_true', help='Run only the configured noise level')
    args = parser.parse_args()
    try:
        metrics = run_demo(args.output, SimulationConfig(seed=args.seed, noise=args.noise),
                           args.components, args.clusters, not args.skip_noise_sweep)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f'ramanspec: {error}\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
