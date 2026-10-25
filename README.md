# ghloc.py
# a rewrite of https://github.com/subtle-byte/ghloc in python and uv
# created using bubblewrap with opencode with glm 5.3 flash max, while listening to sakanaction - Adapt
```
uv run rewrite.py [-c] [-m MATCH]      # was cmd/ghloc:
#   counts LOC of the current directory,
#   -c prints the table to console,
#   no flags starts a local web UI for it
uv run rewrite.py serve                # was cmd/server: HTTP API for GitHub repos
```
# all this, just so I don't install go on my system, lol
# this includes just putting the web ui as literal encoding in there
