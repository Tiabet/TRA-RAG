"""Public multi-hop entry point; delegates to the shared multi-hop runner."""
import asyncio
from multihop_runner import parse_args, run_multihop

if __name__ == '__main__':
    asyncio.run(run_multihop(parse_args()))
