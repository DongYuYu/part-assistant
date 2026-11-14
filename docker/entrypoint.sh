#!/bin/sh
# Container entrypoint: the first argument picks what to run.
#
#   prepare          download datasheets, create the demo database (only if missing),
#                    chunk, download the retrieval models, embed. Safe to run again.
#   ask "question"   ask the agent (needs ANTHROPIC_API_KEY, costs money)
#   mcp              MCP server over stdin/stdout
#   eval [options]   end-to-end golden-set eval (costs money)
#   anything else    run it as a command, e.g. bash
set -eu

case "${1:-}" in
  prepare)
    python -m part_assistant.download
    if [ ! -f data/processed/factory.db ]; then
      python -m part_assistant.seed  # never on restart: it would delete your tickets
    fi
    python -m part_assistant.chunk
    python -c "
from part_assistant.chunk import load_chunks
from part_assistant.retrieve import Embedder, Reranker
e = Embedder(); e.embed_chunks(load_chunks()); Reranker()
print('embedded', e.n_embedded, 'new chunks')"
    echo "ready"
    ;;
  ask)
    shift
    exec python -m part_assistant.agent "$@"
    ;;
  mcp)
    exec python -m part_assistant.mcp_server
    ;;
  eval)
    shift
    exec python -m part_assistant.eval_agent "$@"
    ;;
  "")
    echo "usage: prepare | ask \"question\" | mcp | eval [options] | <command>" >&2
    exit 2
    ;;
  *)
    exec "$@"
    ;;
esac
