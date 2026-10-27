"""Public multi-hop entry point; preserves the original experiment runner."""
import asyncio
from test_pipeline_paths_hint_expansion import parse_args, run_test

if __name__ == '__main__':
    asyncio.run(run_test(parse_args()))
